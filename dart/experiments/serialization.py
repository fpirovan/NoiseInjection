import io
import os
import zipfile

import stable_baselines3
import torch
from stable_baselines3.common.save_util import recursive_setattr, recursive_getattr, json_to_data, data_to_json, \
    open_path, get_device
from stable_baselines3.common.utils import check_for_correct_spaces


def save(model, path, exclude=None, include=None):
    """
    Save all the attributes of the object and the model parameters in a zip-file.

    :param path: path to the file where the rl agent should be saved
    :param exclude: name of parameters that should be excluded in addition to the default ones
    :param include: name of parameters that might be excluded but should be included anyway
    """
    # copy parameter list so we don't mutate the original dict
    data = model.__dict__.copy()

    # exclude is union of specified parameters (if any) and standard exclusions
    if exclude is None:
        exclude = []
    exclude = set(exclude).union(model._excluded_save_params())

    # do not exclude params if they are specifically included
    if include is not None:
        exclude = exclude.difference(include)

    state_dicts_names, torch_variable_names = model._get_torch_save_params()
    all_pytorch_variables = state_dicts_names + torch_variable_names
    for torch_var in all_pytorch_variables:
        # we need to get only the name of the top most module as we'll remove that
        var_name = torch_var.split(".")[0]
        # any params that are in the save vars must not be saved by data
        exclude.add(var_name)

    # remove parameter entries of parameters which are to be excluded
    for param_name in exclude:
        data.pop(param_name, None)

    # build dict of torch variables
    pytorch_variables = None
    if torch_variable_names is not None:
        pytorch_variables = {}
        for name in torch_variable_names:
            attr = recursive_getattr(model, name)
            pytorch_variables[name] = attr

    # build dict of state_dicts
    params_to_save = model.get_parameters()

    save_to_zip_file(path, data=data, params=params_to_save, pytorch_variables=pytorch_variables)


def load(cls, path, env=None, device="auto", custom_objects=None, **kwargs):
    """
    Load the model from a zip-file.

    :param path: path to the file (or a file-like) where to load the agent from
    :param env: the new environment to run the loaded model on
    :param device: Device on which the code should run
    :param custom_objects: Dictionary of objects to replace upon loading
    :param kwargs: extra arguments to change the model when loading
    """
    data, params, pytorch_variables = load_from_zip_file(path, device=device, custom_objects=custom_objects)

    # remove stored device information and replace with ours
    if "policy_kwargs" in data:
        if "device" in data["policy_kwargs"]:
            del data["policy_kwargs"]["device"]

    if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
        raise ValueError(
            f"The specified policy kwargs do not equal the stored policy kwargs."
            f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
        )

    if "observation_space" not in data or "action_space" not in data:
        raise KeyError("The observation_space and action_space were not given, can't verify new environments")

    if env is not None:
        # wrap first if needed
        env = cls._wrap_env(env, data["verbose"])
        # check if given env is valid
        check_for_correct_spaces(env, data["observation_space"], data["action_space"])
    else:
        # use stored env, if one exists. If not, continue as is (can be used for predict)
        if "env" in data:
            env = data["env"]

    model = cls(
        policy=data["policy_class"],
        env=env,
        device=device,
        _init_setup_model=False
    )

    # load parameters
    model.__dict__.update(data)
    model.__dict__.update(kwargs)
    model._setup_model()

    # put state_dicts back in place
    model.set_parameters(params, exact_match=True, device=device)

    # put other pytorch variables back in place
    if pytorch_variables is not None:
        for name in pytorch_variables:
            recursive_setattr(model, name, pytorch_variables[name])

    # sample gSDE exploration matrix, so it uses the right device
    if model.use_sde:
        model.policy.reset_noise()

    return model


def save_to_zip_file(save_path, data=None, params=None, pytorch_variables=None, verbose=0):
    """
    Save model data to a zip archive.

    :param save_path: Where to store the model.
    :param data: Class parameters being stored (non-PyTorch variables)
    :param params: Model parameters being stored expected to contain an entry for every state_dict
    :param pytorch_variables: Other PyTorch variables expected to contain name and value of the variable
    :param verbose: Verbosity level, 0 means only warnings, 2 means debug information
    """
    save_path = open_path(save_path, "w", verbose=0, suffix="zip")
    # data/params can be None, so do not
    # try to serialize them blindly
    if data is not None:
        serialized_data = data_to_json(data)

    # create a zip-archive and write our objects there.
    with zipfile.ZipFile(save_path, mode="w") as archive:
        # do not try to save "None" elements
        if data is not None:
            archive.writestr("data", serialized_data)
        if pytorch_variables is not None:
            with archive.open("pytorch_variables.pth", mode="w") as pytorch_variables_file:
                torch.save(pytorch_variables, pytorch_variables_file, _use_new_zipfile_serialization=False)
        if params is not None:
            for file_name, dict_ in params.items():
                with archive.open(file_name + ".pth", mode="w") as param_file:
                    torch.save(dict_, param_file, _use_new_zipfile_serialization=False)

        # save metadata: library version when file was saved
        archive.writestr("_stable_baselines3_version", stable_baselines3.__version__)


def load_from_zip_file(load_path, load_data=True, custom_objects=None, device="auto", verbose=0):
    """
    Load model data from a .zip archive.

    :param load_path: Where to load the model from
    :param load_data: Whether we should load and return data
    :param custom_objects: Dictionary of objects to replace upon loading
    :param device: Device on which the code should run
    :return: Class parameters, model state_dicts (aka "params", dict of state_dict) and dict of pytorch variables
    """
    load_path = open_path(load_path, "r", verbose=verbose, suffix="zip")

    # set device to cpu if cuda is not available
    device = get_device(device=device)

    # open the zip archive and load data
    try:
        with zipfile.ZipFile(load_path) as archive:
            namelist = archive.namelist()
            # if data or parameters is not in the zip archive, assume they were stored as None
            data = None
            pytorch_variables = None
            params = {}

            if "data" in namelist and load_data:
                # load class parameters that are stored with either JSON or pickle (not PyTorch variables)
                json_data = archive.read("data").decode()
                data = json_to_data(json_data, custom_objects=custom_objects)

            # check for all .pth files and load them using torch.load
            pth_files = [file_name for file_name in namelist if os.path.splitext(file_name)[1] == ".pth"]
            for file_path in pth_files:
                with archive.open(file_path, mode="r") as param_file:
                    # file has to be seekable, but param_file is not, so load in BytesIO first
                    file_content = io.BytesIO()
                    file_content.write(param_file.read())
                    # go to start of file
                    file_content.seek(0)
                    # load the parameters with the right map_location, remove .pth ending with splitext
                    th_object = torch.load(file_content, map_location=device)
                    if file_path == "pytorch_variables.pth" or file_path == "tensors.pth":
                        # PyTorch variables (not state_dicts)
                        pytorch_variables = th_object
                    else:
                        # state dicts
                        params[os.path.splitext(file_path)[0]] = th_object

    except zipfile.BadZipFile:
        # load_path wasn't a zip file
        raise ValueError(f"Error: the file {load_path} wasn't a zip-file")

    return data, params, pytorch_variables
