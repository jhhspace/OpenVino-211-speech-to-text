# -*- coding: utf-8 -*-

# =============================================================================
# Copyright (c) 2020 NVIDIA. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

__all__ = [
    'NeuralGraph',
]

from collections import OrderedDict, defaultdict, namedtuple
from os import path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from ruamel.yaml import YAML
from torch.nn.parallel import DataParallel

from nemo.backends import get_state_dict, load, save, set_state_dict
from nemo.core.neural_factory import DeviceType, OperationMode
from nemo.core.neural_interface import NeuralInterface
from nemo.core.neural_modules import ModuleType, NeuralModule
from nemo.core.neural_types import NeuralPortNameMismatchError, NeuralType, NmTensor
from nemo.package_info import __version__ as nemo_version
from nemo.utils import logging
from nemo.utils.configuration_error import ConfigurationError
from nemo.utils.neural_graph.connection import Connection, StepModulePort
from nemo.utils.neural_graph.graph_inputs import GraphInputs
from nemo.utils.neural_graph.graph_outputs import GraphOutputs

YAML = YAML(typ='safe')


class NeuralGraph(NeuralInterface):
    """
    Neural Graph class stores dynamically defined graphs of connected Neural Modules.
    Once defined, the graph is static, i.e. the connectivity between the nodes is frozen.
    """

    def __init__(self, operation_mode: OperationMode = OperationMode.both, name: Optional[str] = None):
        """
        Constructor. Initializes graph variables.

        Args:
            operation_mode: Graph operation mode, that will be propagated along modules during graph creation.
            [training | eval | both] (DEFAULT: both)
            name: Name of the graph (optional)
        """
        # Initialize the inferface.
        super().__init__()

        # Register graph.
        self._name = self._app_state.register_graph(self, name)

        # Store name and operation mode.
        self._operation_mode = operation_mode

        # "Modules" - list of modules constituting "nodes" in a given graph.
        self._modules = {}

        # All tensors produced within this graph (dict of dicts).
        # This stores  "all output tensors" dictionary, where the key is the name of "producer" module,
        # and the value contains a dictionary of all tensors produced by it.
        self._all_tensors = {}

        # "Steps": order of the  execution of modules in a graph.
        self._steps = OrderedDict()

        # Bound inputs.
        self._inputs = GraphInputs()

        # Bound outputs.
        self._outputs = GraphOutputs(self._all_tensors)

        # Flag indicating whether the "default" output ports/tensors will be automatically bound.
        self.default_output_binding = True

        # Set default data loader params.
        self._data_loader_config = {
            "dataset": None,
            "batch_size": 1,
            "shuffle": False,
            "sampler": None,
            "batch_sampler": None,
            "num_workers": 0,
            "collate_fn": None,
            "pin_memory": False,
            "drop_last": False,
            "timeout": 0,
            "worker_init_fn": None,
            "multiprocessing_context": None,
        }

        # Lazy-initialize loader when needed.
        self._data_loader = None

        # Data collected during forward propagation.
        self._forward_data = defaultdict(lambda: {})

        # Initial device: CPU.
        self._pt_device = torch.device("cpu")

    def __call__(self, **kwargs):
        """
        This method "nests" one existing neural graph into another one.
        Also checks if all inputs were provided and properly connects them.

        Args:
            kwargs: keyword arguments containing dictionary of (input_port_name, port_content).
        """
        # Test operation modes of the nested graphs.
        outer_mode = self._app_state.active_graph.operation_mode
        inner_mode = self.operation_mode

        if inner_mode == OperationMode.evaluation and outer_mode == OperationMode.training:
            raise TypeError("Cannot nest 'inference' graph into 'training'")

        if inner_mode == OperationMode.training and outer_mode == OperationMode.evaluation:
            raise TypeError("Cannot nest 'training' graph into 'inference'")

        if inner_mode == OperationMode.training and outer_mode == OperationMode.both:
            raise TypeError("Cannot nest 'training' graph into 'both'")

        if inner_mode == OperationMode.evaluation and outer_mode == OperationMode.both:
            raise TypeError("Cannot nest 'inference' graph into 'both'")

        # Check inputs: iterate through all inputs passed to the "self".
        for port_name, port_content in kwargs.items():
            # Make sure that passed arguments correspond to input port names.
            if port_name not in self.input_ports.keys():
                raise NeuralPortNameMismatchError(port_name)

        # "Nest" this graph into an active graph.
        results = self._app_state.active_graph.__nest(self, kwargs)

        # Return output tensors.
        return results

    def __nest(self, inner_graph: 'NeuralGraph', inner_graph_args):
        """
        Method nests (copies) a graph: modules, steps, topology (tensors).

        Args:
            inner_graph: Graph to be copied (will be "nested" in this (self) graph).
            inner_graph_args: inputs passed to the graph call.
        """
        # Remember the number of "already present steps".
        step_bump = len(self.steps)

        # "Copy" the modules from nested graph.
        for key, module in inner_graph.modules.items():
            # Check if module with that name already exists.
            # TODO: Uncomment when we will refactor all examples so training/validation graphs won't be added
            # to the "default" graph.
            # if key in self._modules.keys():
            #    raise KeyError("Neural Graph already contains a module named {}".format(module.name))
            self._modules[key] = module

        # Next we should copy the topography - i.e. produce "real" copies of tensors.
        # In fact, instead of copying, we will produce them, following:
        # - the execution order defined in "steps"
        # - connectivity defined in tensor' consumers-ports
        # (so the same logic that will be used in graph deserialization)

        # So let us first serialize the connections of the nested graph.
        # Create a list: (producer.port -> consumer.port)
        inner_connections = []
        for tensors in inner_graph.tensors.values():
            for t in tensors.values():
                inner_connections.extend(t.connections())

        # We need to disable the binding of "defeault" ports on per module basis - we will "manually" produce
        # them only for ports that are already indicated as the "bound" ones in the inner graph.
        self.default_output_binding = False

        # Now "copy" graph execution order and topology by actually executing each step of the nested graph.
        for step_number, module_name in inner_graph.steps.items():
            # Both module and step will be added by the modules' call().

            # Get the module.
            module = inner_graph._modules[module_name]

            # Produce list of arguments that will be passed to a given modules.
            module_args = {}
            # Do it by:
            # - harvesing input port names of a given module,
            # - checking if the input was not bound (in the inner graph),
            # - checking if we have already tensors leading to that input (in outer graph).
            for input_port_name in module.input_ports.keys():
                # Check if this port was bound in the inner graph.
                key = inner_graph.inputs.has_binding(step_number, input_port_name)
                # If so, then we must pass whatever was passed to that port in the list of arguments.
                if key is not None:
                    module_args[input_port_name] = inner_graph_args[key]
                    # As a result, the "module" call() will bind this input!
                    continue

                # Else: find a tensor that should be passed to the given module's input.
                # Search for producer/port that we should use.
                for connection in inner_connections:
                    if (
                        connection.consumer.step_number == step_number
                        and connection.consumer.module_name == module_name
                        and connection.consumer.port_name == input_port_name
                    ):
                        # Got the connection!
                        bumped_step = connection.producer.step_number + step_bump
                        # producer_name = connection.producer.module_name
                        producer_port_name = connection.producer.port_name
                        break

                # Now, the tensor is already produced in outer (i.e. this) graph!
                module_args[input_port_name] = self.tensors[bumped_step][producer_port_name]

            # Ok, now we have all keyword arguments. We can call() the module.
            # This will collect all the produced output tensors and add them to this graph.
            module(**module_args)

        # At that point we have all modules, steps and tensors added to outer (self) graph.
        # Now we have to prepare the outputs.

        # This part is different from Neural Module.
        # Now the goal is NOT to create NEW "tensors", but to return the BOUND ones!
        # Still, those must be bound in the outer (active) graph, but using port names from the inner (nested) graph.

        # Get list of "the adequate output tensors".
        output_tensors = {}
        # Iterate through outputs of the inner graph.
        for key, tensor in inner_graph.output_tensors.items():
            # Find the tensors within this (outer) graph that are outputs by the same producer-port.
            bumped_step = tensor.producer_step_number + step_bump
            # producer_name = tensor.producer_name
            producer_port_name = tensor.name
            # Get adequate tensor from "outer graph" (self).
            output_tensors[key] = self.tensors[bumped_step][producer_port_name]

        if len(output_tensors) == 1:
            # Return a single tensor.
            key = list(output_tensors)[0]
            results = output_tensors[key]

            # Bind the "default" output ports of the inner graph as "default" output ports of this graph.
            # Call the bind() method of bound_outputs directly, as we already have the tensors in our graph.
            # But: Use output port name of the inner graph!
            self.outputs.bind([results], [key])

        else:
            # Create a named tuple type enabling to access outputs by attributes (e.g. out.x).
            output_class_name = f'{self.__class__.__name__}Output'
            result_type = namedtuple(typename=output_class_name, field_names=output_tensors.keys())

            # Return the bound output tensors.
            results = result_type(*output_tensors.values())

            # Bind the "default" output ports of the inner graph as "default" output ports of this graph.
            # Call the bind() method of bound_outputs directly, as we already have the tensors in our graph.
            # But: Use output port name of the inner graph!
            self.outputs.bind(output_tensors.values(), output_tensors.keys())

        # Ok, now we can turn automatic binding on.
        self.default_output_binding = True

        # Return the results.
        return results

    def record_step(self, module: NeuralModule):
        """
        Records the operation (the module to be executed) on a list.

        Args:
            module: Neural modules added to a given graph.

        Returns:
            Step number.
        """
        # The solution allows loops in the graph.
        # This also means that module with that name can already be present in the graph.
        if module.name in self._modules.keys():
            # Check if this is the same module.
            if self._modules[module.name] is not module:
                raise KeyError("Neural Graph already contains a different module with name `{}`!".format(module.name))

        else:
            # Add module to list of modules.
            self._modules[module.name] = module

        # Add step - store the module name.
        step_number = len(self._steps)
        self._steps[step_number] = module.name

        # Return the current step number.
        return step_number

    @property
    def step_number(self) -> int:
        """
        Returns:
            The current step number.
        """
        return len(self._steps) - 1

    def bind_outputs(self, tensors_list: Union[NmTensor, List[NmTensor]]):
        """
        Binds the output tensors.

        Args:
            tensors_list: A single tensor OR a List of tensors to be bound.
        """
        # Handle both single port and lists of ports to be bound.
        if type(tensors_list) is not list:
            tensors_list = [tensors_list]

        # Add tensors to list of list of tensors.
        for tensor in tensors_list:
            # Add tensor to "all" tensors dictionary.
            step_number = tensor.producer_step_number
            if step_number not in self._all_tensors.keys():
                self._all_tensors[step_number] = {}

            port_name = tensor.name
            # Add tensor.
            self._all_tensors[step_number][port_name] = tensor

        # Bind the tensors as graph outputs.
        if self.default_output_binding:
            self.outputs.bind(tensors_list)

    @property
    def inputs(self) -> GraphInputs:
        """
        Returns:
            Graph input.
        """
        return self._inputs

    @property
    def input_ports(self) -> Dict[str, NeuralType]:
        """
        Returns definitions of graph input ports (dict of Neural Types).

        .. note::
            This method actually returns an immutable  dictionary with port types (like Neural Modules).
            In order to get access to actual graph inputs please call the inputs() method.

        Returns:
            Graph input ports definitions.
        """
        return self._inputs.definitions

    @property
    def outputs(self) -> GraphOutputs:
        """
        Returns graph outputs.

        Returns:
            Graph outputs.
        """
        return self._outputs

    @property
    def output_ports(self) -> Dict[str, NeuralType]:
        """
        Returns definitions of module output ports (dict of Neural Types).

        .. note::
            This method actually returns an immutable dictionary with port types (like Neural Modules).
            In order to get access to actual graph outpus please call the outputs() method.

        Returns:
            Graph output ports definitions.
            
        """
        return self._outputs.definitions

    @property
    def output_tensors(self) -> Dict[str, NmTensor]:
        """
        Returns:
            Fraph output tensors.
        """
        return self._outputs.tensors

    @property
    def modules(self) -> Dict[str, NeuralModule]:
        """ Returns modules. """
        return self._modules

    def __getitem__(self, key) -> NeuralModule:
        """ Returns module given its name (name of the variable).

            Args:
                key: Name of the variable.
            
            Raises:
                KeyError: Neural Graph doesn't contain a module with a given name (key).
        """
        if key not in self._modules.keys():
            raise KeyError("Neural Graph doesn't contain a module named {}".format(key))
        return self._modules[key]

    def __len__(self) -> int:
        """
        Returns:
            The number of modules (vertices) in a given graph.
        """
        return len(self._modules)

    @property
    def steps(self) -> Dict[int, str]:
        """
        Returns:
            Dictionary [steps_number, module_name]
        """
        return self._steps

    @property
    def tensors(self):
        """
        Property returning a (double) dictionary of all output tensors.

        Returns:
            Dictionary of tensors in the format [module_name][output_port_name].
         """
        return self._all_tensors

    @property
    def tensor_list(self) -> List[NmTensor]:
        """
        Property returning output tensors by extracting them on the fly from the bound outputs.

        Returns:
            List of tensors.
        """
        tensor_list = []
        # Get tensors by acessing the producer-ports.
        for tensors_per_module in self._all_tensors.values():
            for tensor in tensors_per_module.values():
                # Add it to the list.
                tensor_list.append(tensor)
        # Return the result.
        return tensor_list

    @property
    def operation_mode(self) -> OperationMode:
        """
        Returns:
            Operation mode.
        """
        return self._operation_mode

    def __enter__(self) -> 'NeuralGraph':
        """ 
        Activates this graph.

        Returns:
            The graph object.
        """
        self._app_state.active_graph = self
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """
        Deactivates the current graph.
        """
        self._app_state.active_graph = None

    def activate(self):
        """ 
        Activates this graph.
        """
        self._app_state.active_graph = self

    def deactivate(self):
        """
        Deactivates the current graph.
        """
        self._app_state.active_graph = None

    def export_to_config(self, config_file: str):
        """
        Exports the neural graph to a file.

        Args:
            config_file: Name (and path) of the config file (YML) to be written to.
        """
        # Greate an absolute path.
        abs_path_file = path.expanduser(config_file)

        # Serialize the graph.
        to_export = self.serialize()

        # All parameters are ok, let's export.
        with open(abs_path_file, 'w') as outfile:
            YAML.dump(to_export, outfile)

        logging.info(
            "Configuration of graph `{}` ({}) exported to '{}'".format(self.name, type(self).__name__, abs_path_file)
        )

    def serialize(self) -> Dict[str, Any]:
        """
        Method serializes the whole graph.

        Returns:
            Dictionary containing description of the whole graph.
        """
        # Create a dictionary representing the serialized object.
        serialized_graph = {}

        # Add "header" with module "specification".
        serialized_graph["header"] = self.__serialize_header()

        # Add modules.
        serialized_graph["modules"] = self.__serialize_modules()

        # Add steps.
        serialized_graph["steps"] = self.__serialize_steps()

        # Add connectinos.
        serialized_graph["connections"] = self.__serialize_connections()

        # Serialize graph (bound) inputs.
        serialized_graph["inputs"] = self._inputs.serialize()

        # Serialize graph (bound) outputs.
        serialized_graph["outputs"] = self._outputs.serialize()

        # Return the dictionary.
        return serialized_graph

    def __serialize_header(self) -> Dict[str, Any]:
        """
        Private method responsible for serializing the graph header.

        Returns:
            Dictionary containing description of the whole graph.
        """
        # Generate full_spec of the class.
        full_spec = str(self.__module__) + "." + str(self.__class__.__qualname__)
        header = {"nemo_core_version": nemo_version, "full_spec": full_spec}
        # Add operation mode.
        if self._operation_mode == OperationMode.training:
            header["operation_mode"] = "training"
        elif self._operation_mode == OperationMode.evaluation:
            header["operation_mode"] = "inference"
        else:
            header["operation_mode"] = "both"
        # Return header.
        return header

    def __serialize_modules(self) -> Dict[str, Any]:
        """
        Private method responsible for serializing the modules present in the graph.

        Returns:
            Dictionary containing description of all graph modules.
        """
        serialized_modules = {}
        for name, module in self._modules.items():
            serialized_modules[name] = module.serialize()
        return serialized_modules

    def __serialize_steps(self):
        """
        Private method responsible for serializing the steps (order of module executions).

        Returns:
            Dictionary containing description of the steps.
        """
        serialized_steps = {}
        for no, module_name in self._steps.items():
            serialized_steps[no] = module_name
        return serialized_steps

    def __serialize_connections(self) -> Dict[str, Any]:
        """
        Private method responsible for serializing the connections in the graph.

        Returns:
            List containing "connections" between modules.
        """
        serialized_connections = []
        # Iterate through "tensor modules".
        for tensors in self._all_tensors.values():
            # Iterate through "tensor output ports".
            for tensor in tensors.values():
                # "Transform" tensor to the list of connections.
                for c in tensor.connections():
                    # Serialize!
                    source = str(c.producer.step_number) + "." + c.producer.module_name + "." + c.producer.port_name
                    target = str(c.consumer.step_number) + "." + c.consumer.module_name + "." + c.consumer.port_name
                    ntype_str = str(tensor.ntype)
                    serialized_connections.append(source + "->" + target + " | " + ntype_str)
        return serialized_connections

    @classmethod
    def import_from_config(
        cls,
        config_file: str,
        reuse_existing_modules: bool = False,
        overwrite_params: Dict[str, Any] = {},
        name: Optional[str] = None,
    ) -> 'NeuralGraph':
        """
        Class method importing the neural graph from the configuration file.
        Raises an ImportError exception when config file is invalid.

        Args:
            config_file: path (absolute or relative) and name of the config file (YML)
            reuse_existing_modules: If the modules with (name, type, init_params) are already created, import will
            connect to them instead of creating new instances.
            overwrite_params: Dictionary containing parameters that will be added to or overwrite (!) the default
            parameters loaded from the configuration file
            name: Name of the new graph (optional, DEFAULT: NONE)
        Returns:
            Instance of the created NeuralGraph object.
        """
        logging.info("Loading configuration of a new Neural Graph from the `{}` file".format(config_file))

        # Validate the content of the configuration file (its header).
        loaded_config = cls.__validate_config_file(config_file)
        # TODO: overwrite params?

        # "Deserialize" the graph.
        new_graph = cls.deserialize(loaded_config, reuse_existing_modules, name)

        # Return the object.
        return new_graph

    @classmethod
    def __validate_config_file(cls, config_file: str):
        """
        Class method validating whether the config file has a proper content (sections, specification etc.).
        Raises an ImportError exception when config file is invalid or
        incompatible (when called from a particular class).

        Args:
            config_file: path (absolute or relative) and name of the config file (YML)
        Returns:
            A loaded configuration file (dictionary).
        """
        # Greate an absolute path.
        abs_path_file = path.expanduser(config_file)

        # Open the config file.
        with open(abs_path_file, 'r') as stream:
            loaded_config = YAML.load(stream)

        # Check sections.
        for section_name in ["header", "modules", "steps", "connections", "inputs", "outputs"]:
            if section_name not in loaded_config.keys():
                raise ImportError(
                    "The loaded config `{}` doesn't contain the required `{}` section".format(
                        config_file, section_name
                    )
                )

        # Parse the "full specification".
        spec_list = loaded_config["header"]["full_spec"].split(".")

        # Check if config contains definition of Neural Graph.
        if spec_list[-1] != "NeuralGraph":
            txt = "The loaded file `{}` contains configuration of ".format(config_file)
            txt = txt + "`{}` thus cannot be used for instantiation of Neural Graph".format(spec_list[-1])
            raise ImportError(txt)

        # Success - return the loaded configuration.
        return loaded_config

    @classmethod
    def deserialize(
        cls, configuration: Dict[str, Any], reuse_existing_modules: bool = False, name: Optional[str] = None
    ) -> 'NeuralGraph':
        """
        Class method creating a graph instance by deserializing the provided configuratino.

        Args:
            configuration: Dictionary containing serialized graph.
            reuse_existing_modules: If the modules with (name, type, init_params) are already created, import will
            connect to them instead of creating new instances.
        Returns:
            Instance of the created NeuralGraph object.
        """
        # Deserialize header and get object class.
        operation_mode = cls.__deserialize_header(configuration["header"])

        # Create the graph instance.
        new_graph = NeuralGraph(operation_mode=operation_mode, name=name)
        logging.info(
            "Instantiated a new Neural Graph named `{}` with mode `{}`".format(
                new_graph.name, new_graph.operation_mode
            )
        )
        # Deserialize modules.
        modules = new_graph.__deserialize_modules(configuration["modules"], reuse_existing_modules)

        # Deserialize steps.
        steps = new_graph.__deserialize_steps(configuration["steps"])

        # Deserialize the connections between modules.
        connections = new_graph.__deserialize_connections(configuration["connections"], modules)

        # Deserialize input bindings - return it in an external entity.
        inputs = GraphInputs.deserialize(configuration["inputs"], modules)

        # Deserialize "manual" output bindings.
        new_graph._outputs.deserialize(configuration["outputs"], modules)

        # Now we have to execute the graph, following the steps and connections.
        new_graph.__execute_and_create_tensors(steps, modules, connections, inputs)

        # Return the graph instance.
        return new_graph

    @classmethod
    def __deserialize_header(cls, serialized_header: Dict[str, Any]):
        """
        Private class method deserializing the header and extracts the general information.

        Args:
            serialized_header: Dictionary containing graph header.
        Returns:
            Operation mode.
        """
        # Parse the "full specification" - do not need that now.
        # spec_list = serialized_header["full_spec"].split(".")

        # Get operation mode.
        if serialized_header["operation_mode"] == "training":
            operation_mode = OperationMode.training
        elif serialized_header["operation_mode"] == "inference":
            operation_mode = OperationMode.evaluation
        else:
            operation_mode = OperationMode.both

        # Return the mode.
        return operation_mode

    def __deserialize_modules(self, serialized_modules: Dict[str, Any], reuse_existing_modules: bool):
        """
        Private method deserializing the modules present in the graph.

        Args:
            serialized_modules: Dictionary containing graph modules.
            reuse_existing_modules: If True, won create a new module when a module with a given name exists.

        Returns:
            Dictionary of modules.

        Raises:
            KeyError: A module with name already exists (if reuse_existing_modules is set to False).
        """
        modules = {}
        for name, module_params in serialized_modules.items():
            # Check if module already exists.
            if self._app_state.modules.has(name):
                # Check if we can reuse the existing modules.
                if reuse_existing_modules:
                    modules[name] = self._app_state.modules[name]
                else:
                    raise KeyError("A module with name `{}` already exists!".format(name))
            else:
                # Ok, create a new module.
                modules[name] = NeuralModule.deserialize(module_params)
        # Ok, done.
        return modules

    def __deserialize_steps(self, serialized_steps: Dict[str, Any]):
        """
        Private method deserializing the steps (order of module executions).

        Args:
            serialized_steps: Dictionary containing serialized steps.
        Returns:
            Odered dict with steps.
        """
        steps = OrderedDict()
        for i in range(len(serialized_steps)):
            steps[i] = serialized_steps[i]
        # Ok, done.
        return steps

    def __deserialize_connections(self, serialized_connections: Dict[str, Any], modules: Dict[str, NeuralModule]):
        """
        Private method deserializing the connections in the graph.

        Args:
            serialized_steps: Dictionary containing serialized connections.
            modules: List of modules.
        Returns:
            List of connections, in a format enabling graph traversing.
        """
        connections = []
        # Deserialize connections one by one.
        for c in serialized_connections:
            # Deserialize!
            [producer, consumer_type] = c.split("->")
            [consumer, ntype_str] = consumer_type.split(" | ")
            [producer_step, producer_name, producer_port_name] = producer.split(".")
            [consumer_step, consumer_name, consumer_port_name] = consumer.split(".")
            producer_mp = StepModulePort(int(producer_step), producer_name, producer_port_name)
            consumer_mp = StepModulePort(int(consumer_step), consumer_name, consumer_port_name)
            # Get tensor type.
            ntype = modules[producer_name].output_ports[producer_port_name]
            # Validate if neural type is ok.
            assert ntype_str == str(ntype)

            # Add connection.
            connections.append(Connection(producer_mp, consumer_mp, ntype))
        # Ok, done.
        return connections

    def __execute_and_create_tensors(self, steps, modules, connections, inputs):
        """
        Method creates (internal) tensors of the graph by executing it following the order and using
        the provided connections and inputs.

        Args:
            steps: List of steps to be executed.
            modules: List of modules.
            connections: List of connections.
            inputs: List of "bound inputs"
        """
        # Activate this graph, so all the tensors will be added to this !
        self.activate()

        # We need to disable the binding of "defeault" ports on per module basis.
        # We will "manually" produce (e.g. deserialize) them outside of this function.
        self.default_output_binding = False

        # Now "copy" graph execution order and topology by actually executing each step of the nested graph.
        for step, module_name in steps.items():
            # Both module and step will be added by the modules' call().

            # Get the module.
            module = modules[module_name]

            # Produce list of arguments that will be passed to a given module.
            module_args = {}
            # Do it by:
            # - harvesing input port names of a given module,
            # - checking if the input was not bound (in the inner graph),
            # - checking if we have already tensors leading to that input (in outer graph).
            for input_port_name in module.input_ports.keys():
                # Check if this port was bound in the inner graph.
                key = inputs.has_binding(step, input_port_name)

                # import pdb;pdb.set_trace()
                # If so, then we must pass the binding!
                if key is not None:
                    # Copy the port "definition" (i.e. is NeuralType) using the same port name.
                    self.inputs[key] = inputs[key]

                    # Pass this object to module input argument.
                    module_args[input_port_name] = self.inputs[key]

                # Else: find a tensor that should be passed to the given module's input.
                else:
                    # Search for producer/port that we should use.
                    for connection in connections:
                        if (
                            connection.consumer.step_number == step
                            and connection.consumer.module_name == module_name
                            and connection.consumer.port_name == input_port_name
                        ):
                            # Got the connection!
                            producer_step_number = connection.producer.step_number
                            # producer_name = connection.producer.module_name
                            producer_port_name = connection.producer.port_name
                            break
                    # Now, the tensor is already produced in outer (i.e. this) graph!
                    module_args[input_port_name] = self.tensors[producer_step_number][producer_port_name]
                # End: for

            # Ok, now we have all keyword arguments. We can call() the module.
            # This will collect all the produced output tensors and add them to this graph.
            module(**module_args)

        # At that point we have all modules, steps and tensors added to outer (self) graph.
        # Now we have to prepare the outputs.

        # Deactivate graph.
        self.deactivate()

        # Ok, now we can turn automatic binding on.
        self.default_output_binding = True

    def summary(self) -> str:
        """ 
        Returns:
            A nice, full graph summary.
        """
        # Line "decorator".
        desc = "\n" + 113 * '=' + "\n"
        # 1. general information.
        desc += "The `{}` Neural Graph [{}, ".format(self.name, self.operation_mode)
        if self.is_complete:
            desc += " COMPLETE,"
        else:
            desc += " INCOMPLETE,"
        if self.is_trainable:
            desc += " TRAINABLE]:\n"
        else:
            desc += " NON-TRAINABLE]:\n"

        # 2. modules.
        desc += " * Modules ({}):\n".format(len(self._modules))
        for key, module in self._modules.items():
            if module.type == ModuleType.trainable and module.is_frozen():
                desc += "    * `{}` ({}) [FROZEN]\n".format(key, type(module).__name__)
            else:
                desc += "    * `{}` ({})\n".format(key, type(module).__name__)

        # 3. steps.
        desc += " * Steps ({}):\n".format(len(self._steps))
        for num, module in self._steps.items():
            desc += "    {}. {}\n".format(num, module)

        # 4. connections.
        connections = self.__serialize_connections()
        desc += " * Connections ({}):\n".format(len(connections))
        # if len(connections) == 0:
        #    desc += "    -\n"
        for connection in connections:
            desc += "    * {}\n".format(connection)

        # 5. graph (bound) inputs.
        inputs = self._inputs.serialize()
        desc += " * Graph Inputs ({}):\n".format(len(inputs))
        # if len(inputs) == 0:
        #    desc += "    -\n"
        for input in inputs:
            desc += "    * {}\n".format(input)

        # 6. graph (bound) outputs.
        outputs = self._outputs.serialize()
        desc += " * Graph Outputs ({}, {}):\n".format(len(outputs["mappings"]), outputs["type"])
        # if len(outputs) == 0:
        #    desc += "    -\n"
        for output in outputs["mappings"]:
            desc += "    * {}\n".format(output)
        # Line "decorator".
        desc += 113 * '='

        # Return the result.
        return desc

    def freeze(self, module_names: Optional[List[str]] = None):
        """
        A method that freezes the weights of the trainable modules in a graph.

        Args:
            module_names: List of modules to be frozen (Optional). If passed, all modules will be unfrozen.
        Raises:
            KeyError: If name of the module won't be recognized.
        """
        # Work on all modules.
        if module_names is None:
            module_names = self._modules.keys()

        # Iterate through modules one by one.
        for name in module_names:
            if name not in self._modules.keys():
                raise KeyError("Module `{}` not present in the `{}` graph".format(name, self.name))
            # Check module type.
            module = self._modules[name]
            if module.type == ModuleType.trainable:
                # Freeze weights of the module.
                module.freeze()
            else:
                logging.debug("Module `{}` is not trainable so cannot be frozen".format(name))

    def unfreeze(self, module_names: Optional[List[str]] = None):
        """
        Unfreezes weights of the trainable modules in a graph.

        Args:
            module_names: List of modules to be unfrozen (Optional). If not passed, all modules will be unfrozen.
        Raises:
            KeyError: If name of the module won't be recognized.
        """
        # Work on all modules.
        if module_names is None:
            module_names = self._modules.keys()

        # Iterate through modules one by one.
        for name in module_names:
            if name not in self._modules.keys():
                raise KeyError("Module `{}` not present in the `{}` graph".format(name, self.name))
            # Check module type.
            module = self._modules[name]
            if module.type == ModuleType.trainable:
                # Unfreeze weights of the module.
                module.unfreeze()
            else:
                logging.debug("Module `{}` is not trainable so cannot be unfrozen".format(name))

    def save_to(self, filename: str, module_names: Optional[List[str]] = None):
        """
        Saves the state of trainable modules in the graph to a checkpoint file.

        Args:
            filename (string): Name of the file where the checkpoint will be saved.
            module_names: List of modules to be frozen (Optional). If passed, all modules will be saved.
        Raises:
            KeyError: If name of the module won't be recognized.
        """
        # Work on all modules.
        if module_names is None:
            module_names = self._modules.keys()

        # Prepare the "graph checkpoint".
        chkpt = {"header": {"nemo_core_version": nemo_version, "name": self.name}, "modules": {}}

        log_str = ''
        # Iterate through the modules one by one.
        for name in module_names:
            if name not in self._modules.keys():
                raise KeyError("Module `{}` not present in the `{}` graph".format(name, self.name))
            # Check module type.
            module = self._modules[name]
            if module.type == ModuleType.trainable:
                # Get module state_dict().
                chkpt["modules"][name] = get_state_dict(module)
                log_str += "  * Module '{}' ({}) params saved \n".format(module.name, type(module).__name__)
            else:
                logging.debug("Module `{}` is not trainable so cannot be saved".format(name))

        # Save checkpoint.
        save(chkpt, filename)
        log_str = "Saved  the '{}' graph to a checkpoint `{}`:\n".format(self.name, filename) + log_str
        logging.info(log_str)

    def restore_from(self, filename: str, module_names: Optional[List[str]] = None):
        """
        Restores the state of trainable modules in the graph from a checkpoint file.

        Args:
            filename (string): Name of the checkpoint to be restored from.
            module_names: List of modules to be frozen (Optional). If passed, all modules will be restored.
        Raises:
            KeyError: If name of the module won't be recognized.
        """
        # Work on all modules.
        if module_names is None:
            module_names = self._modules.keys()

        # Load the checkpoint.
        chkpt = load(filename)

        log_str = "Loading modules constituting the '{}' graph from the `{}` checkpoint :\n".format(
            chkpt["header"]["name"], filename
        )

        warning = False
        # Iterate through the modules one by one.
        for name in module_names:
            try:
                # Get module.
                module = self._modules[name]
                if module.type == ModuleType.trainable:
                    # Restore module weights
                    set_state_dict(module, chkpt["modules"][name])
                    log_str += "  * Module '{}' ({}) params loaded\n".format(module.name, type(module).__name__)
            except KeyError:
                log_str += "  ! Module '{}' params not found in checkpoint\n".format(name)
                warning = True

        # Log results.
        if warning:
            logging.warning(log_str)
        else:
            logging.info(log_str)

    @property
    def is_complete(self) -> bool:
        """
        Property checks if graph is "complete", which means that the graph has:
            * exactly one DataLayer
            * zero bound input ports

        Returns:
            True or false.
        """
        # We assume the first modules is DL.
        dl = self.modules[self.steps[0]]
        if dl.type != ModuleType.datalayer:
            return False

        # We assume there is only one DL in the whole graph.
        has_datalayer = False
        # Iterate through the modules one by one.
        for module in self._modules.values():
            # Get module type.
            if module.type == ModuleType.datalayer:
                if has_datalayer:
                    # More than one DL is not acceptable.
                    return False
                else:
                    has_datalayer = True

        # Now check the ports.
        if len(self._inputs) != 0:
            return False

        # Else:
        return True

    @property
    def is_trainable(self) -> bool:
        """
        Property checks if graph is "trainable", which means that the graph:
            * is in eval mode
            * has at least one Loss Neural Modules

        Returns:
            True or false.
        """
        # Check if gradients are collected in forward pass.
        if self.operation_mode == OperationMode.evaluation:
            return False

        # Iterate through the modules one by one.
        for module in self._modules.values():
            # Get module type.
            if module.type == ModuleType.loss:
                return True

        # No loss modules - cannot train the graph using back-propagation.
        return False

    def to(self, device_type: DeviceType, use_dataparallel: bool = False):
        """ 
        Moves the all the (trainable) modules belonging to a given graph to indicated device.

        """
        logging.info("Moving module(s) to {}".format(device_type))

        # Check device.
        if device_type == DeviceType.CPU:
            self._pt_device = torch.device("cpu")

        elif device_type == DeviceType.GPU:
            if not torch.cuda.is_available():
                raise ConfigurationError("Coudn't use GPUs as there is no CUDA installed")

            if torch.cuda.device_count() == 1:
                # Cannot use data parallel.
                if use_dataparallel:
                    raise ConfigurationError("Coudn't use Data Parallel as there is only one GPU device found")

            # Using cuda -- all devices.
            self._pt_device = torch.device("cuda")

        else:  # DeviceType.AllGpu : distributed data parallel.
            if not torch.cuda.is_available():
                raise ConfigurationError("Coudn't use Distributed Data Parallel as there is no CUDA installed")

            # Each worker will use a single gpu.
            use_dataparallel = False
            self._pt_device = torch.device("cuda")

        if use_dataparallel:
            logging.info("Using Data Parallelization on {} GPUs!".format(torch.cuda.device_count()))
            logging.warning("Data Parallel is currently under development, so use if it with caution")

        # Work on all modules.
        module_names = self._modules.keys()

        # Iterate through the modules one by one.
        for name in module_names:
            # Get module.
            module = self._modules[name]
            if module.type == ModuleType.trainable:
                # Check if we want to use data parallel.
                if use_dataparallel:
                    # Skip if the user explicitly said the module shouldn't be parallelized.
                    if not hasattr(module, 'skip_in_data_parallel') or not module.skip_in_data_parallel:
                        # Parallize model.
                        module = DataParallel(module)

                # Mode to device.
                self._modules[name] = module.to(self._pt_device)

    def configure_data_loader(
        self,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        multiprocessing_context=None,
    ):
        """
        Method updates the default parameters of the DataLoader used by a given neural graph.
        For the details on the function/meanings of the arguments, please refer to:
        https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        """

        # Override all old values.
        self._data_loader_config = {
            "dataset": None,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "sampler": sampler,
            "batch_sampler": batch_sampler,
            "num_workers": num_workers,
            "collate_fn": collate_fn,
            "pin_memory": pin_memory,
            "drop_last": drop_last,
            "timeout": timeout,
            "worker_init_fn": worker_init_fn,
            "multiprocessing_context": multiprocessing_context,
        }
        # If loader is set - reset it.
        self._data_loader = None

    def get_batch(self, yield_dict: bool = False) -> Union[Dict[str, Any], Tuple]:
        """
        Method yields a single batch. Optionally, instantiates the DataLoader object used by a given graph.
            
        Args:
            yield_dict: Flag used to yield a tuple or a dict - depending on the user needs
            (DEFAULT: False, i.e. yielding NamedTuples)
        Returns:
            Depending on the mode: a batch in the form of a singe namedtuple or of a dictionary.
        """
        # If loader is not set - set it.
        if self._data_loader is None:
            # Check graph.
            if not self.is_complete:
                raise ConfigurationError("Cannot get a batch as graph is incomplete!")
            # Get datalayer/dataset.
            dl = self.modules[self.steps[0]]
            # Make sure that this is DataLayer/dataset instance.
            if isinstance(dl, torch.utils.data.Dataset):
                self._data_loader_config["dataset"] = dl
            else:
                self._data_loader_config["dataset"] = dl.dataset

            self._data_loader = torch.utils.data.DataLoader(**self._data_loader_config)

        # Return tuple or dict - depending on the user needs.
        if yield_dict:
            # Fetch a batch - in the form of a dict.
            for batch in self._data_loader:
                if len(dl.output_ports.keys()) == 1:
                    # Dict with a single key:value pair.
                    batch = batch.to(self._pt_device) if isinstance(batch, torch.Tensor) else batch
                    yield {list(dl.output_ports.keys())[0]: batch}
                else:
                    # Yield a dictionary.
                    batch = [elem.to(self._pt_device) if isinstance(elem, torch.Tensor) else elem for elem in batch]
                    yield dict(zip(dl.output_ports.keys(), batch))

        else:
            # Create a tuple class - if required.
            if len(dl.output_ports.keys()) > 1:
                # Create a named tuple type enabling to access outputs by attributes (e.g. out.x).
                output_class_name = f'{dl.__class__.__name__}Output'
                result_type = namedtuple(typename=output_class_name, field_names=dl.output_ports.keys())

            # Fetch a batch - in the form of a tuple.
            for batch in self._data_loader:
                if len(dl.output_ports.keys()) == 1:
                    # Return a single object.
                    batch = batch.to(self._pt_device) if isinstance(batch, torch.Tensor) else batch
                    yield batch
                else:
                    # Return a tuple.
                    batch = [elem.to(self._pt_device) if isinstance(elem, torch.Tensor) else elem for elem in batch]
                    yield result_type(*batch)

    def __forward(self, inputs: Dict[str, Any], connections: Dict[str, Any]):
        """
        Graph forward method. In the execution it follows the order of the modules defined in steps and passes data
        between modules following the connectivity stored in NmTensors.
        """
        # Execute modules one by one.
        for step_number in range(len(self._steps)):
            # If graph is complete - we already fetched data from module 0 (DL)
            if self.is_complete and step_number == 0:
                # So lets skip it.
                continue

            # Get the module.
            module_name = self._steps[step_number]
            module = self._modules[module_name]

            # Change the module's operation mode.
            module.operation_mode = self.operation_mode

            # Produce list of arguments that will be passed to a given modules.
            module_args = {}
            # Do it by:
            # - harvesing input port names of a given module,
            # - checking if the input was not bound (in the inner graph),
            # - checking if we have already tensors leading to that input (in outer graph).
            for input_port_name in module.input_ports.keys():
                # Check if this port was bound in the inner graph.
                key = self.inputs.has_binding(step_number, input_port_name)
                # If so, then we must pass whatever was passed to that port in the list of arguments.
                if key is not None:
                    module_args[input_port_name] = inputs[key]
                    continue

                # Else: find a tensor that should be passed to the given module's input.
                producer_step = -1
                # Search for producer/port that we should use.
                for connection in connections:
                    if (
                        connection.consumer.step_number == step_number
                        and connection.consumer.module_name == module_name
                        and connection.consumer.port_name == input_port_name
                    ):
                        # Got the connection!
                        producer_step = connection.producer.step_number
                        # producer_name = connection.producer.module_name
                        producer_port_name = connection.producer.port_name
                        break
                # Make sure we found the producer.
                if producer_step == -1:
                    # Check if this port is not optional.
                    if not module.input_ports[input_port_name].optional:
                        err = "Couldn't find the producer of the {} input to the module {} called in step {}".format(
                            input_port_name, module_name, step_number
                        )
                        raise ValueError(err)
                    # If it is optional and not provided - simply do nothing...
                else:
                    # Add this to the argument
                    module_args[input_port_name] = self._forward_data[producer_step][producer_port_name]

            # Ok, now we have all keyword arguments. We can call() the module.
            module_outputs = module(force_pt=True, **module_args)

            # Collect all the produced output tensors and add them to this graph.
            output_names = list(module.output_ports.keys())

            if len(output_names) == 1:
                # Handle the case of a single data produced.
                self._forward_data[step_number][output_names[0]] = module_outputs
            else:
                # Compare module_outputs with the output port definitions.
                if len(output_names) != len(module_outputs):
                    err = "Invalid number of outputs produced by the module "
                    err += "{} - expected: `{}`, received: `{}`".format(module_name, output_names, len(module_outputs))
                    raise ValueError(err)

                # Add them to the passed data.
                for key, value in zip(output_names, module_outputs):
                    self._forward_data[step_number][key] = value

        # Ok, now return only the bound outputs.
        if len(self.outputs) == 1:
            # Return a single object.
            smp = self.outputs[list(self.outputs.keys())[0]].producer_step_module_port
            return self._forward_data[smp.step_number][smp.port_name]
        elif len(self.outputs) > 1:
            # Return a tuple.
            output_class_name = f'{self.__class__.__name__}Output'
            result_type = namedtuple(typename=output_class_name, field_names=self.outputs.keys())
            # Prepare list values.
            values = []
            for output in self.outputs.values():
                smp = output.producer_step_module_port
                values.append(self._forward_data[smp.step_number][smp.port_name])
            # Return the object
            return result_type(*values)
        else:  # (==0)
            # Generally this should not happen!
            return

    def forward(self, *args: Optional[Tuple], **kwargs: Optional[Dict[str, Any]]):
        """
        Graph forward method. In the execution it follows the order of the modules defined in steps and passes data
        between modules following the connectivity stored in NmTensors.

        Accepts a batch (passed as a tuple in args) OR a list of named arguments (as kwargs).

        Use-case 1:
        
        .. code-block:: python

            # Retrieve batch as a tuple and pass it to forward() as tuple.
            for batch in training_graph.get_batch():
                training_graph.forward(batch)

        Use-case 2:

        .. code-block:: python

            # Retrieve batch as a tuple and pass it to forward() as list of named arguments.
            for batch in training_graph.get_batch():
                training_graph.forward(input1=batch.input1, input2=batch.input2)


        Use-case 3:

        .. code-block:: python

            # Retrieve batch as dictionary and pass it to forward() as list of named arguments.
            for batch in training_graph.get_batch(yield_dict=True):
                training_graph.forward(**batch)

        Args:
            args: A tuple object (a batch) with all required inputs (optional)
            kwargs: a dictionary containing all required inputs (optional)

        Returns:
            A tuple object containing all graph outputs
        """

        # Get list of argument names.
        if self.is_complete:
            # Use dataset definitions.
            dl = self.modules[self.steps[0]]
            input_names = list(dl.output_ports.keys())
        else:
            input_names = list(self._inputs.keys())

        # Work on args or kwargs - depending on input_dict settings.
        if len(args) > 0:
            # Operate on args, i.e. a single tuple or a single "object" (1 input port to graph).
            if len(args) == 1:
                # Check if it is a tuple.
                if isinstance(args[0], tuple) and hasattr(args[0], "_fields"):
                    inputs = args[0]._asdict()
                else:
                    # There is only one input - use the input_names[0] as key.
                    inputs = {input_names[0]: args[0]}
            else:
                err = "Invalid argument passed to `graph forward()`"
                err += " - expected: a tuple, received: `{}`".format(args)
                raise ValueError(err)
        else:
            # Operate on named arguments.
            inputs = kwargs

        # Compare inputs with the desired inputs.
        if len(input_names) != len(inputs) or sorted(input_names) != sorted(inputs.keys()):
            err = "Invalid list of arguments passed to the `graph forward()`"
            err += " - expected: `{}`, received: `{}`".format(input_names, inputs.keys())
            raise ValueError(err)

        # Copy inputs to an adequate (module.output->value) structure.
        if self._forward_data is not None:
            self._forward_data = None
        self._forward_data = defaultdict(lambda: {})

        if self.is_complete:
            # Treat all the inputs as "DL outputs".
            for key, value in inputs.items():
                self._forward_data[0][key] = value

        # Create a list of connenctions: (producer.port -> consumer.port)
        connections = []
        for tensors in self.tensors.values():
            for t in tensors.values():
                connections.extend(t.connections())

        if self.operation_mode == OperationMode.evaluation:
            # Perform forward - without collecting of the gradients.
            with torch.no_grad():
                return self.__forward(inputs, connections)
        else:
            # Perform forward - and collect all the gradients.
            return self.__forward(inputs, connections)

    def backward(self, losses: List["Tensor"] = []):
        """
        Backward pass through the graph. Optionally the method accepts list of losses to backpropagate from.
        If not provided, will collect all output of all loss modules in the graph and backpropagate from them.

        Args:
            losses: list of losses to backpropagate from (OPTIONAL, DEFAULT: [])

        Raises:
            ConfigurationError if graph is non-trainable.
        """
        # Check if we can run backwards at all.
        if not self.is_trainable:
            raise ConfigurationError(
                "Cannot run backward propagation as the graph `{}`` is non-trainable".format(self.name)
            )

        losses_to_backpropagate = []
        # If losses are passed - use those during backpropagation.
        if len(losses) != 0:
            losses_to_backpropagate = losses
        else:
            # Else: collect outputs of all Loss NMs.
            for step_number in range(len(self._steps)):

                # Get the module.
                module_name = self._steps[step_number]
                module = self._modules[module_name]

                # Collect module outputs.
                # Assumption: loss modules return only loss.
                if module.type == ModuleType.loss:
                    for output_port_names in module.output_ports.keys():
                        losses_to_backpropagate.append(self._forward_data[step_number][output_port_names])

        # Estimate the total number of backward passes (one from each tensor).
        total_passes = len(losses_to_backpropagate)

        # All but the last call to backward should have the retain_graph=True option.
        pass_counter = 0
        for loss in losses_to_backpropagate:
            pass_counter += 1
            if pass_counter == total_passes:
                # Last pass.
                loss.backward()
            else:
                # "Other pass."
                loss.backward(retain_graph=True)

    def parameters(self, recurse: bool = True):
        """
        Args:
            recurse (bool): if True, then yields parameters of graph modules and all their submodules.
        Returns:
            An iterator over parameter of all trainable modules.
        """
        for module_name in self._modules:
            # Get module.
            module = self._modules[module_name]
            if module.type == ModuleType.trainable:
                # Yield it parameters.
                for _, param in module.named_parameters(recurse=recurse):
                    yield param

    def named_parameters(self, recurse: bool = True):
        """
        Args:
            recurse (bool): if True, then yields parameters of graph modules and all their submodules.
        Returns:
            An iterator ovar all named parameters of all trainable components.
        """
        for module_name in self._modules:
            # Get module.
            module = self._modules[module_name]
            if module.type == ModuleType.trainable:
                # Yield it parameters.
                for name, param in module.named_parameters(recurse=recurse):
                    yield name, param
