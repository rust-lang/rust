# -*clang- Python -*-

import os
import platform
import re
import subprocess

import lit.formats
import lit.util
from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'BenchEnzyme'

# testFormat: The test format to use to interpret tests.
#
# For now we require '&&' between commands, until they get globally killed and
# the test runner updated.
execute_external = platform.system() != 'Windows'
config.test_format = lit.formats.ShTest(execute_external)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.make']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.enzyme_obj_root, 'test')

#ToolSubst('%lli', FindTool('lli'), post='.', extra_args=lli_args),

# Tweak the PATH to include the tools dir and the scripts dir.
base_paths = [config.llvm_tools_dir, config.environment['PATH']]
path = os.path.pathsep.join(base_paths) # + config.extra_paths)
config.environment['PATH'] = path

path = os.path.pathsep.join((config.llvm_libs_dir,
                              config.environment.get('LD_LIBRARY_PATH','')))
config.environment['LD_LIBRARY_PATH'] = path

#tools = ['opt', 'lli', 'clang', 'clang++']
#llvm_config.add_tool_substitutions(tools, config.llvm_tools_dir)

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
