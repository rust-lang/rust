# -*clang- Python -*-

import os
import platform
import re
import subprocess

import lit.formats
import lit.util

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'Enzyme-c-tests'

# testFormat: The test format to use to interpret tests.
#
# For now we require '&&' between commands, until they get globally killed and
# the test runner updated.
execute_external = platform.system() != 'Windows'
config.test_format = lit.formats.ShTest(execute_external)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.test']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)


print("config.test_source_root is " + config.test_source_root)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.enzyme_obj_root, 'functional_tests_c')

# Tweak the PATH to include the tools dir and the scripts dir.
base_paths = [config.llvm_tools_dir, config.environment['PATH']]
path = os.path.pathsep.join(base_paths) # + config.extra_paths)
config.environment['PATH'] = path

path = os.path.pathsep.join((config.llvm_libs_dir,
                              config.environment.get('LD_LIBRARY_PATH','')))
config.environment['LD_LIBRARY_PATH'] = path

#export TAPIR_PREFIX=./../../llvm/build4
#export ENZYME_PLUGIN=./../build4/Enzyme/LLVMEnzyme-7.so

#config.environment['TAPIR_PREFIX'] = config.llvm_src_root
config.environment['ENZYME_PLUGIN'] = config.enzyme_obj_root +'/Enzyme/LLVMEnzyme-' + config.llvm_ver + config.llvm_shlib_ext

# opt knows whether it is compiled with -DNDEBUG.
import subprocess
try:
    opt_cmd = subprocess.Popen([os.path.join(config.llvm_tools_dir, 'opt'), '-version'],
                           stdout = subprocess.PIPE,
                           env=config.environment)
except OSError:
    print("Could not find opt in " + config.llvm_tools_dir)
    exit(42)

if re.search(r'with assertions', opt_cmd.stdout.read().decode('ascii')):
    config.available_features.add('asserts')
opt_cmd.wait()

try:
    llvm_config_cmd = subprocess.Popen([os.path.join(
                                        config.llvm_tools_dir,
                                        'llvm-config'),
                                        '--targets-built'],
                                       stdout = subprocess.PIPE,
                                       env=config.environment)
except OSError:
    print("Could not find llvm-config in " + config.llvm_tools_dir)
    exit(42)
