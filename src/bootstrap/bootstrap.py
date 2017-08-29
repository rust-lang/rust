# Copyright 2015-2016 The Rust Project Developers. See the COPYRIGHT
# file at the top-level directory of this distribution and at
# http://rust-lang.org/COPYRIGHT.
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

from __future__ import print_function
import argparse
import contextlib
import datetime
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

from time import time


def get(url, path, verbose=False):
    suffix = '.sha256'
    sha_url = url + suffix
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as sha_file:
        sha_path = sha_file.name

    try:
        download(sha_path, sha_url, False, verbose)
        if os.path.exists(path):
            if verify(path, sha_path, False):
                if verbose:
                    print("using already-download file", path)
                return
            else:
                if verbose:
                    print("ignoring already-download file",
                          path, "due to failed verification")
                os.unlink(path)
        download(temp_path, url, True, verbose)
        if not verify(temp_path, sha_path, verbose):
            raise RuntimeError("failed verification")
        if verbose:
            print("moving {} to {}".format(temp_path, path))
        shutil.move(temp_path, path)
    finally:
        delete_if_present(sha_path, verbose)
        delete_if_present(temp_path, verbose)


def delete_if_present(path, verbose):
    """Remove the given file if present"""
    if os.path.isfile(path):
        if verbose:
            print("removing", path)
        os.unlink(path)


def download(path, url, probably_big, verbose):
    for _ in range(0, 4):
        try:
            _download(path, url, probably_big, verbose, True)
            return
        except RuntimeError:
            print("\nspurious failure, trying again")
    _download(path, url, probably_big, verbose, False)


def _download(path, url, probably_big, verbose, exception):
    if probably_big or verbose:
        print("downloading {}".format(url))
    # see http://serverfault.com/questions/301128/how-to-download
    if sys.platform == 'win32':
        run(["PowerShell.exe", "/nologo", "-Command",
             "(New-Object System.Net.WebClient)"
             ".DownloadFile('{}', '{}')".format(url, path)],
            verbose=verbose,
            exception=exception)
    else:
        if probably_big or verbose:
            option = "-#"
        else:
            option = "-s"
        run(["curl", option, "--retry", "3", "-Sf", "-o", path, url],
            verbose=verbose,
            exception=exception)


def verify(path, sha_path, verbose):
    """Check if the sha256 sum of the given path is valid"""
    if verbose:
        print("verifying", path)
    with open(path, "rb") as source:
        found = hashlib.sha256(source.read()).hexdigest()
    with open(sha_path, "r") as sha256sum:
        expected = sha256sum.readline().split()[0]
    verified = found == expected
    if not verified:
        print("invalid checksum:\n"
              "    found:    {}\n"
              "    expected: {}".format(found, expected))
    return verified


def unpack(tarball, dst, verbose=False, match=None):
    """Unpack the given tarball file"""
    print("extracting", tarball)
    fname = os.path.basename(tarball).replace(".tar.gz", "")
    with contextlib.closing(tarfile.open(tarball)) as tar:
        for member in tar.getnames():
            if "/" not in member:
                continue
            name = member.replace(fname + "/", "", 1)
            if match is not None and not name.startswith(match):
                continue
            name = name[len(match) + 1:]

            dst_path = os.path.join(dst, name)
            if verbose:
                print("  extracting", member)
            tar.extract(member, dst)
            src_path = os.path.join(dst, member)
            if os.path.isdir(src_path) and os.path.exists(dst_path):
                continue
            shutil.move(src_path, dst_path)
    shutil.rmtree(os.path.join(dst, fname))


def run(args, verbose=False, exception=False, **kwargs):
    """Run a child program in a new process"""
    if verbose:
        print("running: " + ' '.join(args))
    sys.stdout.flush()
    # Use Popen here instead of call() as it apparently allows powershell on
    # Windows to not lock up waiting for input presumably.
    ret = subprocess.Popen(args, **kwargs)
    code = ret.wait()
    if code != 0:
        err = "failed to run: " + ' '.join(args)
        if verbose or exception:
            raise RuntimeError(err)
        sys.exit(err)


def stage0_data(rust_root):
    """Build a dictionary from stage0.txt"""
    nightlies = os.path.join(rust_root, "src/stage0.txt")
    with open(nightlies, 'r') as nightlies:
        lines = [line.rstrip() for line in nightlies
                 if not line.startswith("#")]
        return dict([line.split(": ", 1) for line in lines if line])


def format_build_time(duration):
    """Return a nicer format for build time

    >>> format_build_time('300')
    '0:05:00'
    """
    return str(datetime.timedelta(seconds=int(duration)))


def default_build_triple():
    """Build triple as in LLVM"""
    default_encoding = sys.getdefaultencoding()
    try:
        ostype = subprocess.check_output(
            ['uname', '-s']).strip().decode(default_encoding)
        cputype = subprocess.check_output(
            ['uname', '-m']).strip().decode(default_encoding)
    except (subprocess.CalledProcessError, OSError):
        if sys.platform == 'win32':
            return 'x86_64-pc-windows-msvc'
        err = "uname not found"
        sys.exit(err)

    # The goal here is to come up with the same triple as LLVM would,
    # at least for the subset of platforms we're willing to target.
    ostype_mapper = {
        'Bitrig': 'unknown-bitrig',
        'Darwin': 'apple-darwin',
        'DragonFly': 'unknown-dragonfly',
        'FreeBSD': 'unknown-freebsd',
        'Haiku': 'unknown-haiku',
        'NetBSD': 'unknown-netbsd',
        'OpenBSD': 'unknown-openbsd'
    }

    # Consider the direct transformation first and then the special cases
    if ostype in ostype_mapper:
        ostype = ostype_mapper[ostype]
    elif ostype == 'Linux':
        os_from_sp = subprocess.check_output(
            ['uname', '-o']).strip().decode(default_encoding)
        if os_from_sp == 'Android':
            ostype = 'linux-android'
        else:
            ostype = 'unknown-linux-gnu'
    elif ostype == 'SunOS':
        ostype = 'sun-solaris'
        # On Solaris, uname -m will return a machine classification instead
        # of a cpu type, so uname -p is recommended instead.  However, the
        # output from that option is too generic for our purposes (it will
        # always emit 'i386' on x86/amd64 systems).  As such, isainfo -k
        # must be used instead.
        try:
            cputype = subprocess.check_output(
                ['isainfo', '-k']).strip().decode(default_encoding)
        except (subprocess.CalledProcessError, OSError):
            err = "isainfo not found"
            sys.exit(err)
    elif ostype.startswith('MINGW'):
        # msys' `uname` does not print gcc configuration, but prints msys
        # configuration. so we cannot believe `uname -m`:
        # msys1 is always i686 and msys2 is always x86_64.
        # instead, msys defines $MSYSTEM which is MINGW32 on i686 and
        # MINGW64 on x86_64.
        ostype = 'pc-windows-gnu'
        cputype = 'i686'
        if os.environ.get('MSYSTEM') == 'MINGW64':
            cputype = 'x86_64'
    elif ostype.startswith('MSYS'):
        ostype = 'pc-windows-gnu'
    elif ostype.startswith('CYGWIN_NT'):
        cputype = 'i686'
        if ostype.endswith('WOW64'):
            cputype = 'x86_64'
        ostype = 'pc-windows-gnu'
    else:
        err = "unknown OS type: {}".format(ostype)
        sys.exit(err)

    cputype_mapper = {
        'BePC': 'i686',
        'aarch64': 'aarch64',
        'amd64': 'x86_64',
        'arm64': 'aarch64',
        'i386': 'i686',
        'i486': 'i686',
        'i686': 'i686',
        'i786': 'i686',
        'powerpc': 'powerpc',
        'powerpc64': 'powerpc64',
        'powerpc64le': 'powerpc64le',
        'ppc': 'powerpc',
        'ppc64': 'powerpc64',
        'ppc64le': 'powerpc64le',
        's390x': 's390x',
        'x64': 'x86_64',
        'x86': 'i686',
        'x86-64': 'x86_64',
        'x86_64': 'x86_64'
    }

    # Consider the direct transformation first and then the special cases
    if cputype in cputype_mapper:
        cputype = cputype_mapper[cputype]
    elif cputype in {'xscale', 'arm'}:
        cputype = 'arm'
        if ostype == 'linux-android':
            ostype = 'linux-androideabi'
    elif cputype == 'armv6l':
        cputype = 'arm'
        if ostype == 'linux-android':
            ostype = 'linux-androideabi'
        else:
            ostype += 'eabihf'
    elif cputype in {'armv7l', 'armv8l'}:
        cputype = 'armv7'
        if ostype == 'linux-android':
            ostype = 'linux-androideabi'
        else:
            ostype += 'eabihf'
    elif cputype == 'mips':
        if sys.byteorder == 'big':
            cputype = 'mips'
        elif sys.byteorder == 'little':
            cputype = 'mipsel'
        else:
            raise ValueError("unknown byteorder: {}".format(sys.byteorder))
    elif cputype == 'mips64':
        if sys.byteorder == 'big':
            cputype = 'mips64'
        elif sys.byteorder == 'little':
            cputype = 'mips64el'
        else:
            raise ValueError('unknown byteorder: {}'.format(sys.byteorder))
        # only the n64 ABI is supported, indicate it
        ostype += 'abi64'
    elif cputype == 'sparcv9':
        pass
    else:
        err = "unknown cpu type: {}".format(cputype)
        sys.exit(err)

    return "{}-{}".format(cputype, ostype)

class RustBuild(object):
    """Provide all the methods required to build Rust"""
    def __init__(self):
        self.cargo_channel = ''
        self.date = ''
        self._download_url = 'https://static.rust-lang.org'
        self.rustc_channel = ''
        self.build = ''
        self.build_dir = os.path.join(os.getcwd(), "build")
        self.clean = False
        self.config_toml = ''
        self.printed = False
        self.rust_root = os.path.abspath(os.path.join(__file__, '../../..'))
        self.use_locked_deps = ''
        self.use_vendored_sources = ''
        self.verbose = False

    def download_stage0(self):
        """Fetch the build system for Rust, written in Rust

        This method will build a cache directory, then it will fetch the
        tarball which has the stage0 compiler used to then bootstrap the Rust
        compiler itself.

        Each downloaded tarball is extracted, after that, the script
        will move all the content to the right place.
        """
        rustc_channel = self.rustc_channel
        cargo_channel = self.cargo_channel

        if self.rustc().startswith(self.bin_root()) and \
                (not os.path.exists(self.rustc()) or
                 self.program_out_of_date(self.rustc_stamp())):
            self.print_what_bootstrap_means()
            if os.path.exists(self.bin_root()):
                shutil.rmtree(self.bin_root())
            filename = "rust-std-{}-{}.tar.gz".format(
                rustc_channel, self.build)
            pattern = "rust-std-{}".format(self.build)
            self._download_stage0_helper(filename, pattern)

            filename = "rustc-{}-{}.tar.gz".format(rustc_channel, self.build)
            self._download_stage0_helper(filename, "rustc")
            self.fix_executable("{}/bin/rustc".format(self.bin_root()))
            self.fix_executable("{}/bin/rustdoc".format(self.bin_root()))
            with open(self.rustc_stamp(), 'w') as rust_stamp:
                rust_stamp.write(self.date)

            if "pc-windows-gnu" in self.build:
                filename = "rust-mingw-{}-{}.tar.gz".format(
                    rustc_channel, self.build)
                self._download_stage0_helper(filename, "rust-mingw")

        if self.cargo().startswith(self.bin_root()) and \
                (not os.path.exists(self.cargo()) or
                 self.program_out_of_date(self.cargo_stamp())):
            self.print_what_bootstrap_means()
            filename = "cargo-{}-{}.tar.gz".format(cargo_channel, self.build)
            self._download_stage0_helper(filename, "cargo")
            self.fix_executable("{}/bin/cargo".format(self.bin_root()))
            with open(self.cargo_stamp(), 'w') as cargo_stamp:
                cargo_stamp.write(self.date)

    def _download_stage0_helper(self, filename, pattern):
        cache_dst = os.path.join(self.build_dir, "cache")
        rustc_cache = os.path.join(cache_dst, self.date)
        if not os.path.exists(rustc_cache):
            os.makedirs(rustc_cache)

        url = "{}/dist/{}".format(self._download_url, self.date)
        tarball = os.path.join(rustc_cache, filename)
        if not os.path.exists(tarball):
            get("{}/{}".format(url, filename), tarball, verbose=self.verbose)
        unpack(tarball, self.bin_root(), match=pattern, verbose=self.verbose)

    @staticmethod
    def fix_executable(fname):
        """Modifies the interpreter section of 'fname' to fix the dynamic linker

        This method is only required on NixOS and uses the PatchELF utility to
        change the dynamic linker of ELF executables.

        Please see https://nixos.org/patchelf.html for more information
        """
        default_encoding = sys.getdefaultencoding()
        try:
            ostype = subprocess.check_output(
                ['uname', '-s']).strip().decode(default_encoding)
        except subprocess.CalledProcessError:
            return
        except OSError as reason:
            if getattr(reason, 'winerror', None) is not None:
                return
            raise reason

        if ostype != "Linux":
            return

        if not os.path.exists("/etc/NIXOS"):
            return
        if os.path.exists("/lib"):
            return

        # At this point we're pretty sure the user is running NixOS
        nix_os_msg = "info: you seem to be running NixOS. Attempting to patch"
        print(nix_os_msg, fname)

        try:
            interpreter = subprocess.check_output(
                ["patchelf", "--print-interpreter", fname])
            interpreter = interpreter.strip().decode(default_encoding)
        except subprocess.CalledProcessError as reason:
            print("warning: failed to call patchelf:", reason)
            return

        loader = interpreter.split("/")[-1]

        try:
            ldd_output = subprocess.check_output(
                ['ldd', '/run/current-system/sw/bin/sh'])
            ldd_output = ldd_output.strip().decode(default_encoding)
        except subprocess.CalledProcessError as reason:
            print("warning: unable to call ldd:", reason)
            return

        for line in ldd_output.splitlines():
            libname = line.split()[0]
            if libname.endswith(loader):
                loader_path = libname[:len(libname) - len(loader)]
                break
        else:
            print("warning: unable to find the path to the dynamic linker")
            return

        correct_interpreter = loader_path + loader

        try:
            subprocess.check_output(
                ["patchelf", "--set-interpreter", correct_interpreter, fname])
        except subprocess.CalledProcessError as reason:
            print("warning: failed to call patchelf:", reason)
            return

    def rustc_stamp(self):
        """Return the path for .rustc-stamp

        >>> rb = RustBuild()
        >>> rb.build_dir = "build"
        >>> rb.rustc_stamp() == os.path.join("build", "stage0", ".rustc-stamp")
        True
        """
        return os.path.join(self.bin_root(), '.rustc-stamp')

    def cargo_stamp(self):
        """Return the path for .cargo-stamp

        >>> rb = RustBuild()
        >>> rb.build_dir = "build"
        >>> rb.cargo_stamp() == os.path.join("build", "stage0", ".cargo-stamp")
        True
        """
        return os.path.join(self.bin_root(), '.cargo-stamp')

    def program_out_of_date(self, stamp_path):
        """Check if the given program stamp is out of date"""
        if not os.path.exists(stamp_path) or self.clean:
            return True
        with open(stamp_path, 'r') as stamp:
            return self.date != stamp.read()

    def bin_root(self):
        """Return the binary root directory

        >>> rb = RustBuild()
        >>> rb.build_dir = "build"
        >>> rb.bin_root() == os.path.join("build", "stage0")
        True

        When the 'build' property is given should be a nested directory:

        >>> rb.build = "devel"
        >>> rb.bin_root() == os.path.join("build", "devel", "stage0")
        True
        """
        return os.path.join(self.build_dir, self.build, "stage0")

    def get_toml(self, key):
        """Returns the value of the given key in config.toml, otherwise returns None

        >>> rb = RustBuild()
        >>> rb.config_toml = 'key1 = "value1"\\nkey2 = "value2"'
        >>> rb.get_toml("key2")
        'value2'

        If the key does not exists, the result is None:

        >>> rb.get_toml("key3") == None
        True
        """
        for line in self.config_toml.splitlines():
            match = re.match(r'^{}\s*=(.*)$'.format(key), line)
            if match is not None:
                value = match.group(1)
                return self.get_string(value) or value.strip()
        return None

    def cargo(self):
        """Return config path for cargo"""
        return self.program_config('cargo')

    def rustc(self):
        """Return config path for rustc"""
        return self.program_config('rustc')

    def program_config(self, program):
        """Return config path for the given program

        >>> rb = RustBuild()
        >>> rb.config_toml = 'rustc = "rustc"\\n'
        >>> rb.program_config('rustc')
        'rustc'
        >>> cargo_path = rb.program_config('cargo')
        >>> cargo_path.rstrip(".exe") == os.path.join("/tmp/rust",
        ... "bin", "cargo")
        True
        >>> rb.config_toml = ''
        >>> cargo_path = rb.program_config('cargo')
        >>> cargo_path.rstrip(".exe") == os.path.join(rb.bin_root(),
        ... "bin", "cargo")
        True
        """
        config = self.get_toml(program)
        if config:
            return config
        return os.path.join(self.bin_root(), "bin", "{}{}".format(
            program, self.exe_suffix()))

    @staticmethod
    def get_string(line):
        """Return the value between double quotes

        >>> RustBuild.get_string('    "devel"   ')
        'devel'
        """
        start = line.find('"')
        if start != -1:
            end = start + 1 + line[start + 1:].find('"')
            return line[start + 1:end]
        start = line.find('\'')
        if start != -1:
            end = start + 1 + line[start + 1:].find('\'')
            return line[start + 1:end]
        return None

    @staticmethod
    def exe_suffix():
        """Return a suffix for executables"""
        if sys.platform == 'win32':
            return '.exe'
        return ''

    def print_what_bootstrap_means(self):
        """Prints more information about the build system"""
        if hasattr(self, 'printed'):
            return
        self.printed = True
        if os.path.exists(self.bootstrap_binary()):
            return
        if '--help' not in sys.argv or len(sys.argv) == 1:
            return

        print('info: the build system for Rust is written in Rust, so this')
        print('      script is now going to download a stage0 rust compiler')
        print('      and then compile the build system itself')
        print('')
        print('info: in the meantime you can read more about rustbuild at')
        print('      src/bootstrap/README.md before the download finishes')

    def bootstrap_binary(self):
        """Return the path of the boostrap binary

        >>> rb = RustBuild()
        >>> rb.build_dir = "build"
        >>> rb.bootstrap_binary() == os.path.join("build", "bootstrap",
        ... "debug", "bootstrap")
        True
        """
        return os.path.join(self.build_dir, "bootstrap", "debug", "bootstrap")

    def build_bootstrap(self):
        """Build bootstrap"""
        self.print_what_bootstrap_means()
        build_dir = os.path.join(self.build_dir, "bootstrap")
        if self.clean and os.path.exists(build_dir):
            shutil.rmtree(build_dir)
        env = os.environ.copy()
        env["RUSTC_BOOTSTRAP"] = '1'
        env["CARGO_TARGET_DIR"] = build_dir
        env["RUSTC"] = self.rustc()
        env["LD_LIBRARY_PATH"] = os.path.join(self.bin_root(), "lib") + \
            (os.pathsep + env["LD_LIBRARY_PATH"]) \
            if "LD_LIBRARY_PATH" in env else ""
        env["DYLD_LIBRARY_PATH"] = os.path.join(self.bin_root(), "lib") + \
            (os.pathsep + env["DYLD_LIBRARY_PATH"]) \
            if "DYLD_LIBRARY_PATH" in env else ""
        env["LIBRARY_PATH"] = os.path.join(self.bin_root(), "lib") + \
            (os.pathsep + env["LIBRARY_PATH"]) \
            if "LIBRARY_PATH" in env else ""
        env["PATH"] = os.path.join(self.bin_root(), "bin") + \
            os.pathsep + env["PATH"]
        if not os.path.isfile(self.cargo()):
            raise Exception("no cargo executable found at `{}`".format(
                self.cargo()))
        args = [self.cargo(), "build", "--manifest-path",
                os.path.join(self.rust_root, "src/bootstrap/Cargo.toml")]
        if self.verbose:
            args.append("--verbose")
            if self.verbose > 1:
                args.append("--verbose")
        if self.use_locked_deps:
            args.append("--locked")
        if self.use_vendored_sources:
            args.append("--frozen")
        run(args, env=env, verbose=self.verbose)

    def build_triple(self):
        """Build triple as in LLVM"""
        default_encoding = sys.getdefaultencoding()
        config = self.get_toml('build')
        if config:
            return config
        return default_build_triple()

    def update_submodules(self):
        """Update submodules"""
        if (not os.path.exists(os.path.join(self.rust_root, ".git"))) or \
                self.get_toml('submodules') == "false":
            return
        print('Updating submodules')
        default_encoding = sys.getdefaultencoding()
        run(["git", "submodule", "-q", "sync"], cwd=self.rust_root)
        submodules = [s.split(' ', 1)[1] for s in subprocess.check_output(
            ["git", "config", "--file",
             os.path.join(self.rust_root, ".gitmodules"),
             "--get-regexp", "path"]
        ).decode(default_encoding).splitlines()]
        submodules = [module for module in submodules
                      if not ((module.endswith("llvm") and
                               self.get_toml('llvm-config')) or
                              (module.endswith("jemalloc") and
                               self.get_toml('jemalloc')))]
        run(["git", "submodule", "update",
             "--init", "--recursive"] + submodules,
            cwd=self.rust_root, verbose=self.verbose)
        run(["git", "submodule", "-q", "foreach", "git",
             "reset", "-q", "--hard"],
            cwd=self.rust_root, verbose=self.verbose)
        run(["git", "submodule", "-q", "foreach", "git",
             "clean", "-qdfx"],
            cwd=self.rust_root, verbose=self.verbose)

    def set_dev_environment(self):
        """Set download URL for development environment"""
        self._download_url = 'https://dev-static.rust-lang.org'


def bootstrap():
    """Configure, fetch, build and run the initial bootstrap"""
    parser = argparse.ArgumentParser(description='Build rust')
    parser.add_argument('--config')
    parser.add_argument('--build')
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = [a for a in sys.argv if a != '-h' and a != '--help']
    args, _ = parser.parse_known_args(args)

    # Configure initial bootstrap
    build = RustBuild()
    build.verbose = args.verbose
    build.clean = args.clean

    try:
        with open(args.config or 'config.toml') as config:
            build.config_toml = config.read()
    except:
        pass

    if '\nverbose = 2' in build.config_toml:
        build.verbose = 2
    elif '\nverbose = 1' in build.config_toml:
        build.verbose = 1

    build.use_vendored_sources = '\nvendor = true' in build.config_toml

    build.use_locked_deps = '\nlocked-deps = true' in build.config_toml

    if 'SUDO_USER' in os.environ and not build.use_vendored_sources:
        if os.environ.get('USER') != os.environ['SUDO_USER']:
            build.use_vendored_sources = True
            print('info: looks like you are running this command under `sudo`')
            print('      and so in order to preserve your $HOME this will now')
            print('      use vendored sources by default. Note that if this')
            print('      does not work you should run a normal build first')
            print('      before running a command like `sudo make install`')

    if build.use_vendored_sources:
        if not os.path.exists('.cargo'):
            os.makedirs('.cargo')
        with open('.cargo/config', 'w') as cargo_config:
            cargo_config.write("""
                [source.crates-io]
                replace-with = 'vendored-sources'
                registry = 'https://example.com'

                [source.vendored-sources]
                directory = '{}/src/vendor'
            """.format(build.rust_root))
    else:
        if os.path.exists('.cargo'):
            shutil.rmtree('.cargo')

    data = stage0_data(build.rust_root)
    build.date = data['date']
    build.rustc_channel = data['rustc']
    build.cargo_channel = data['cargo']

    if 'dev' in data:
        build.set_dev_environment()

    build.update_submodules()

    # Fetch/build the bootstrap
    build.build = args.build or build.build_triple()
    build.download_stage0()
    sys.stdout.flush()
    build.build_bootstrap()
    sys.stdout.flush()

    # Run the bootstrap
    args = [build.bootstrap_binary()]
    args.extend(sys.argv[1:])
    env = os.environ.copy()
    env["BUILD"] = build.build
    env["SRC"] = build.rust_root
    env["BOOTSTRAP_PARENT_ID"] = str(os.getpid())
    env["BOOTSTRAP_PYTHON"] = sys.executable
    run(args, env=env, verbose=build.verbose)


def main():
    """Entry point for the bootstrap process"""
    start_time = time()
    help_triggered = (
        '-h' in sys.argv) or ('--help' in sys.argv) or (len(sys.argv) == 1)
    try:
        bootstrap()
        if not help_triggered:
            print("Build completed successfully in {}".format(
                format_build_time(time() - start_time)))
    except (SystemExit, KeyboardInterrupt) as error:
        if hasattr(error, 'code') and isinstance(error.code, int):
            exit_code = error.code
        else:
            exit_code = 1
            print(error)
        if not help_triggered:
            print("Build completed unsuccessfully in {}".format(
                format_build_time(time() - start_time)))
        sys.exit(exit_code)


if __name__ == '__main__':
    main()
