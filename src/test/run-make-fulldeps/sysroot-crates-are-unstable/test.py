import sys
import os
from os import listdir
from os.path import isfile, join
from subprocess import PIPE, Popen


# This is a whitelist of files which are stable crates or simply are not crates,
# we don't check for the instability of these crates as they're all stable!
STABLE_CRATES = ['std', 'alloc', 'core', 'proc_macro',
                 'rsbegin.o', 'rsend.o', 'dllcrt2.o', 'crt2.o', 'clang_rt']


def convert_to_string(s):
    if s.__class__.__name__ == 'bytes':
        return s.decode('utf-8')
    return s


def exec_command(command, to_input=None):
    child = None
    if to_input is None:
        child = Popen(command, stdout=PIPE, stderr=PIPE)
    else:
        child = Popen(command, stdout=PIPE, stderr=PIPE, stdin=PIPE)
    stdout, stderr = child.communicate(input=to_input)
    return (convert_to_string(stdout), convert_to_string(stderr))


def check_lib(lib):
    if lib['name'] in STABLE_CRATES:
        return True
    print('verifying if {} is an unstable crate'.format(lib['name']))
    stdout, stderr = exec_command([os.environ['RUSTC'], '-', '--crate-type', 'rlib',
                                   '--extern', '{}={}'.format(lib['name'], lib['path'])],
                                  to_input='extern crate {};'.format(lib['name']))
    if not 'use of unstable library feature' in '{}{}'.format(stdout, stderr):
        print('crate {} "{}" is not unstable'.format(lib['name'], lib['path']))
        print('{}{}'.format(stdout, stderr))
        print('')
        return False
    return True

# Generate a list of all crates in the sysroot. To do this we list all files in
# rustc's sysroot, look at the filename, strip everything after the `-`, and
# strip the leading `lib` (if present)
def get_all_libs(dir_path):
    return [{ 'path': join(dir_path, f), 'name': f[3:].split('-')[0] }
            for f in listdir(dir_path)
            if isfile(join(dir_path, f)) and f.endswith('.rlib') and f not in STABLE_CRATES]


sysroot = exec_command([os.environ['RUSTC'], '--print', 'sysroot'])[0].replace('\n', '')
libs = get_all_libs(join(sysroot, 'lib/rustlib/{}/lib'.format(os.environ['TARGET'])))

ret = 0
for lib in libs:
    if not check_lib(lib):
        # We continue so users can see all the not unstable crates.
        ret = 1
sys.exit(ret)
