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
import shutil
import subprocess
import sys
import tarfile
import tempfile

from time import time


def get(url, path, verbose=False):
    sha_url = url + ".sha256"
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
    with tempfile.NamedTemporaryFile(suffix=".sha256", delete=False) as sha_file:
        sha_path = sha_file.name

    try:
        download(sha_path, sha_url, False, verbose)
        if os.path.exists(path):
            if verify(path, sha_path, False):
                if verbose:
                    print("using already-download file " + path)
                return
            else:
                if verbose:
                    print("ignoring already-download file " + path + " due to failed verification")
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
    if os.path.isfile(path):
        if verbose:
            print("removing " + path)
        os.unlink(path)


def download(path, url, probably_big, verbose):
    for x in range(0, 4):
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
    if verbose:
        print("verifying " + path)
    with open(path, "rb") as f:
        found = hashlib.sha256(f.read()).hexdigest()
    with open(sha_path, "r") as f:
        expected = f.readline().split()[0]
    verified = found == expected
    if not verified:
        print("invalid checksum:\n"
               "    found:    {}\n"
               "    expected: {}".format(found, expected))
    return verified


def unpack(tarball, dst, verbose=False, match=None):
    print("extracting " + tarball)
    fname = os.path.basename(tarball).replace(".tar.gz", "")
    with contextlib.closing(tarfile.open(tarball)) as tar:
        for p in tar.getnames():
            if "/" not in p:
                continue
            name = p.replace(fname + "/", "", 1)
            if match is not None and not name.startswith(match):
                continue
            name = name[len(match) + 1:]

            fp = os.path.join(dst, name)
            if verbose:
                print("  extracting " + p)
            tar.extract(p, dst)
            tp = os.path.join(dst, p)
            if os.path.isdir(tp) and os.path.exists(fp):
                continue
            shutil.move(tp, fp)
    shutil.rmtree(os.path.join(dst, fname))

def run(args, verbose=False, exception=False):
    if verbose:
        print("running: " + ' '.join(args))
    sys.stdout.flush()
    # Use Popen here instead of call() as it apparently allows powershell on
    # Windows to not lock up waiting for input presumably.
    ret = subprocess.Popen(args)
    code = ret.wait()
    if code != 0:
        err = "failed to run: " + ' '.join(args)
        if verbose or exception:
            raise RuntimeError(err)
        sys.exit(err)

def stage0_data(rust_root):
    nightlies = os.path.join(rust_root, "src/stage0.txt")
    data = {}
    with open(nightlies, 'r') as nightlies:
        for line in nightlies:
            line = line.rstrip()  # Strip newline character, '\n'
            if line.startswith("#") or line == '':
                continue
            a, b = line.split(": ", 1)
            data[a] = b
    return data

def format_build_time(duration):
    return str(datetime.timedelta(seconds=int(duration)))


class RustBuild(object):
    def download_stage0(self):
        cache_dst = os.path.join(self.build_dir, "cache")
        rustc_cache = os.path.join(cache_dst, self.stage0_rustc_date())
        if not os.path.exists(rustc_cache):
            os.makedirs(rustc_cache)

        channel = self.stage0_rustc_channel()

        if self.rustc().startswith(self.bin_root()) and \
                (not os.path.exists(self.rustc()) or self.rustc_out_of_date()):
            self.print_what_it_means_to_bootstrap()
            if os.path.exists(self.bin_root()):
                shutil.rmtree(self.bin_root())
            filename = "rust-std-{}-{}.tar.gz".format(channel, self.build)
            url = "https://static.rust-lang.org/dist/" + self.stage0_rustc_date()
            tarball = os.path.join(rustc_cache, filename)
            if not os.path.exists(tarball):
                get("{}/{}".format(url, filename), tarball, verbose=self.verbose)
            unpack(tarball, self.bin_root(),
                   match="rust-std-" + self.build,
                   verbose=self.verbose)

            filename = "rustc-{}-{}.tar.gz".format(channel, self.build)
            url = "https://static.rust-lang.org/dist/" + self.stage0_rustc_date()
            tarball = os.path.join(rustc_cache, filename)
            if not os.path.exists(tarball):
                get("{}/{}".format(url, filename), tarball, verbose=self.verbose)
            unpack(tarball, self.bin_root(), match="rustc", verbose=self.verbose)
            self.fix_executable(self.bin_root() + "/bin/rustc")
            self.fix_executable(self.bin_root() + "/bin/rustdoc")
            with open(self.rustc_stamp(), 'w') as f:
                f.write(self.stage0_rustc_date())

            if "pc-windows-gnu" in self.build:
                filename = "rust-mingw-{}-{}.tar.gz".format(channel, self.build)
                url = "https://static.rust-lang.org/dist/" + self.stage0_rustc_date()
                tarball = os.path.join(rustc_cache, filename)
                if not os.path.exists(tarball):
                    get("{}/{}".format(url, filename), tarball, verbose=self.verbose)
                unpack(tarball, self.bin_root(), match="rust-mingw", verbose=self.verbose)

        if self.cargo().startswith(self.bin_root()) and \
                (not os.path.exists(self.cargo()) or self.cargo_out_of_date()):
            self.print_what_it_means_to_bootstrap()
            filename = "cargo-{}-{}.tar.gz".format(channel, self.build)
            url = "https://static.rust-lang.org/dist/" + self.stage0_rustc_date()
            tarball = os.path.join(rustc_cache, filename)
            if not os.path.exists(tarball):
                get("{}/{}".format(url, filename), tarball, verbose=self.verbose)
            unpack(tarball, self.bin_root(), match="cargo", verbose=self.verbose)
            self.fix_executable(self.bin_root() + "/bin/cargo")
            with open(self.cargo_stamp(), 'w') as f:
                f.write(self.stage0_rustc_date())

    def fix_executable(self, fname):
        # If we're on NixOS we need to change the path to the dynamic loader

        default_encoding = sys.getdefaultencoding()
        try:
            ostype = subprocess.check_output(['uname', '-s']).strip().decode(default_encoding)
        except (subprocess.CalledProcessError, WindowsError):
            return

        if ostype != "Linux":
            return

        if not os.path.exists("/etc/NIXOS"):
            return
        if os.path.exists("/lib"):
            return

        # At this point we're pretty sure the user is running NixOS
        print("info: you seem to be running NixOS. Attempting to patch " + fname)

        try:
            interpreter = subprocess.check_output(["patchelf", "--print-interpreter", fname])
            interpreter = interpreter.strip().decode(default_encoding)
        except subprocess.CalledProcessError as e:
            print("warning: failed to call patchelf: %s" % e)
            return

        loader = interpreter.split("/")[-1]

        try:
            ldd_output = subprocess.check_output(['ldd', '/run/current-system/sw/bin/sh'])
            ldd_output = ldd_output.strip().decode(default_encoding)
        except subprocess.CalledProcessError as e:
            print("warning: unable to call ldd: %s" % e)
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
            subprocess.check_output(["patchelf", "--set-interpreter", correct_interpreter, fname])
        except subprocess.CalledProcessError as e:
            print("warning: failed to call patchelf: %s" % e)
            return

    def stage0_rustc_date(self):
        return self._rustc_date

    def stage0_rustc_channel(self):
        return self._rustc_channel

    def rustc_stamp(self):
        return os.path.join(self.bin_root(), '.rustc-stamp')

    def cargo_stamp(self):
        return os.path.join(self.bin_root(), '.cargo-stamp')

    def rustc_out_of_date(self):
        if not os.path.exists(self.rustc_stamp()) or self.clean:
            return True
        with open(self.rustc_stamp(), 'r') as f:
            return self.stage0_rustc_date() != f.read()

    def cargo_out_of_date(self):
        if not os.path.exists(self.cargo_stamp()) or self.clean:
            return True
        with open(self.cargo_stamp(), 'r') as f:
            return self.stage0_rustc_date() != f.read()

    def bin_root(self):
        return os.path.join(self.build_dir, self.build, "stage0")

    def get_toml(self, key):
        for line in self.config_toml.splitlines():
            if line.startswith(key + ' ='):
                return self.get_string(line)
        return None

    def get_mk(self, key):
        for line in iter(self.config_mk.splitlines()):
            if line.startswith(key + ' '):
                var = line[line.find(':=') + 2:].strip()
                if var != '':
                    return var
        return None

    def cargo(self):
        config = self.get_toml('cargo')
        if config:
            return config
        config = self.get_mk('CFG_LOCAL_RUST_ROOT')
        if config:
            return config + '/bin/cargo' + self.exe_suffix()
        return os.path.join(self.bin_root(), "bin/cargo" + self.exe_suffix())

    def rustc(self):
        config = self.get_toml('rustc')
        if config:
            return config
        config = self.get_mk('CFG_LOCAL_RUST_ROOT')
        if config:
            return config + '/bin/rustc' + self.exe_suffix()
        return os.path.join(self.bin_root(), "bin/rustc" + self.exe_suffix())

    def get_string(self, line):
        start = line.find('"')
        end = start + 1 + line[start + 1:].find('"')
        return line[start + 1:end]

    def exe_suffix(self):
        if sys.platform == 'win32':
            return '.exe'
        else:
            return ''

    def print_what_it_means_to_bootstrap(self):
        if hasattr(self, 'printed'):
            return
        self.printed = True
        if os.path.exists(self.bootstrap_binary()):
            return
        if not '--help' in sys.argv or len(sys.argv) == 1:
            return

        print('info: the build system for Rust is written in Rust, so this')
        print('      script is now going to download a stage0 rust compiler')
        print('      and then compile the build system itself')
        print('')
        print('info: in the meantime you can read more about rustbuild at')
        print('      src/bootstrap/README.md before the download finishes')

    def bootstrap_binary(self):
        return os.path.join(self.build_dir, "bootstrap/debug/bootstrap")

    def build_bootstrap(self):
        self.print_what_it_means_to_bootstrap()
        build_dir = os.path.join(self.build_dir, "bootstrap")
        if self.clean and os.path.exists(build_dir):
            shutil.rmtree(build_dir)
        env = os.environ.copy()
        env["CARGO_TARGET_DIR"] = build_dir
        env["RUSTC"] = self.rustc()
        env["LD_LIBRARY_PATH"] = os.path.join(self.bin_root(), "lib") + \
                                 (os.pathsep + env["LD_LIBRARY_PATH"]) \
                                 if "LD_LIBRARY_PATH" in env else ""
        env["DYLD_LIBRARY_PATH"] = os.path.join(self.bin_root(), "lib") + \
                                   (os.pathsep + env["DYLD_LIBRARY_PATH"]) \
                                   if "DYLD_LIBRARY_PATH" in env else ""
        env["PATH"] = os.path.join(self.bin_root(), "bin") + \
                      os.pathsep + env["PATH"]
        if not os.path.isfile(self.cargo()):
            raise Exception("no cargo executable found at `%s`" % self.cargo())
        args = [self.cargo(), "build", "--manifest-path",
                os.path.join(self.rust_root, "src/bootstrap/Cargo.toml")]
        if self.use_locked_deps:
            args.append("--locked")
        if self.use_vendored_sources:
            args.append("--frozen")
        self.run(args, env)

    def run(self, args, env):
        proc = subprocess.Popen(args, env=env)
        ret = proc.wait()
        if ret != 0:
            sys.exit(ret)

    def build_triple(self):
        default_encoding = sys.getdefaultencoding()
        config = self.get_toml('build')
        if config:
            return config
        config = self.get_mk('CFG_BUILD')
        if config:
            return config
        try:
            ostype = subprocess.check_output(['uname', '-s']).strip().decode(default_encoding)
            cputype = subprocess.check_output(['uname', '-m']).strip().decode(default_encoding)
        except (subprocess.CalledProcessError, OSError):
            if sys.platform == 'win32':
                return 'x86_64-pc-windows-msvc'
            err = "uname not found"
            if self.verbose:
                raise Exception(err)
            sys.exit(err)

        # The goal here is to come up with the same triple as LLVM would,
        # at least for the subset of platforms we're willing to target.
        if ostype == 'Linux':
            os = subprocess.check_output(['uname', '-o']).strip().decode(default_encoding)
            if os == 'Android':
                ostype = 'linux-android'
            else:
                ostype = 'unknown-linux-gnu'
        elif ostype == 'FreeBSD':
            ostype = 'unknown-freebsd'
        elif ostype == 'DragonFly':
            ostype = 'unknown-dragonfly'
        elif ostype == 'Bitrig':
            ostype = 'unknown-bitrig'
        elif ostype == 'OpenBSD':
            ostype = 'unknown-openbsd'
        elif ostype == 'NetBSD':
            ostype = 'unknown-netbsd'
        elif ostype == 'SunOS':
            ostype = 'sun-solaris'
            # On Solaris, uname -m will return a machine classification instead
            # of a cpu type, so uname -p is recommended instead.  However, the
            # output from that option is too generic for our purposes (it will
            # always emit 'i386' on x86/amd64 systems).  As such, isainfo -k
            # must be used instead.
            try:
                cputype = subprocess.check_output(['isainfo',
                  '-k']).strip().decode(default_encoding)
            except (subprocess.CalledProcessError, OSError):
                err = "isainfo not found"
                if self.verbose:
                    raise Exception(err)
                sys.exit(err)
        elif ostype == 'Darwin':
            ostype = 'apple-darwin'
        elif ostype == 'Haiku':
            ostype = 'unknown-haiku'
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
            err = "unknown OS type: " + ostype
            if self.verbose:
                raise ValueError(err)
            sys.exit(err)

        if cputype in {'i386', 'i486', 'i686', 'i786', 'x86'}:
            cputype = 'i686'
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
        elif cputype in {'aarch64', 'arm64'}:
            cputype = 'aarch64'
        elif cputype == 'mips':
            if sys.byteorder == 'big':
                cputype = 'mips'
            elif sys.byteorder == 'little':
                cputype = 'mipsel'
            else:
                raise ValueError('unknown byteorder: ' + sys.byteorder)
        elif cputype == 'mips64':
            if sys.byteorder == 'big':
                cputype = 'mips64'
            elif sys.byteorder == 'little':
                cputype = 'mips64el'
            else:
                raise ValueError('unknown byteorder: ' + sys.byteorder)
            # only the n64 ABI is supported, indicate it
            ostype += 'abi64'
        elif cputype in {'powerpc', 'ppc'}:
            cputype = 'powerpc'
        elif cputype in {'powerpc64', 'ppc64'}:
            cputype = 'powerpc64'
        elif cputype in {'powerpc64le', 'ppc64le'}:
            cputype = 'powerpc64le'
        elif cputype == 'sparcv9':
            pass
        elif cputype in {'amd64', 'x86_64', 'x86-64', 'x64'}:
            cputype = 'x86_64'
        elif cputype == 's390x':
            cputype = 's390x'
        elif cputype == 'BePC':
            cputype = 'i686'
        else:
            err = "unknown cpu type: " + cputype
            if self.verbose:
                raise ValueError(err)
            sys.exit(err)

        return "{}-{}".format(cputype, ostype)

def bootstrap():
    parser = argparse.ArgumentParser(description='Build rust')
    parser.add_argument('--config')
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')

    args = [a for a in sys.argv if a != '-h' and a != '--help']
    args, _ = parser.parse_known_args(args)

    # Configure initial bootstrap
    rb = RustBuild()
    rb.config_toml = ''
    rb.config_mk = ''
    rb.rust_root = os.path.abspath(os.path.join(__file__, '../../..'))
    rb.build_dir = os.path.join(os.getcwd(), "build")
    rb.verbose = args.verbose
    rb.clean = args.clean

    try:
        with open(args.config or 'config.toml') as config:
            rb.config_toml = config.read()
    except:
        pass
    try:
        rb.config_mk = open('config.mk').read()
    except:
        pass

    rb.use_vendored_sources = '\nvendor = true' in rb.config_toml or \
                              'CFG_ENABLE_VENDOR' in rb.config_mk

    rb.use_locked_deps = '\nlocked-deps = true' in rb.config_toml or \
                         'CFG_ENABLE_LOCKED_DEPS' in rb.config_mk

    if 'SUDO_USER' in os.environ and not rb.use_vendored_sources:
        if os.environ.get('USER') != os.environ['SUDO_USER']:
            rb.use_vendored_sources = True
            print('info: looks like you are running this command under `sudo`')
            print('      and so in order to preserve your $HOME this will now')
            print('      use vendored sources by default. Note that if this')
            print('      does not work you should run a normal build first')
            print('      before running a command like `sudo make install`')

    if rb.use_vendored_sources:
        if not os.path.exists('.cargo'):
            os.makedirs('.cargo')
        with open('.cargo/config','w') as f:
            f.write("""
                [source.crates-io]
                replace-with = 'vendored-sources'
                registry = 'https://example.com'

                [source.vendored-sources]
                directory = '{}/src/vendor'
            """.format(rb.rust_root))
    else:
        if os.path.exists('.cargo'):
            shutil.rmtree('.cargo')

    data = stage0_data(rb.rust_root)
    rb._rustc_channel, rb._rustc_date = data['rustc'].split('-', 1)

    # Fetch/build the bootstrap
    rb.build = rb.build_triple()
    rb.download_stage0()
    sys.stdout.flush()
    rb.build_bootstrap()
    sys.stdout.flush()

    # Run the bootstrap
    args = [rb.bootstrap_binary()]
    args.extend(sys.argv[1:])
    env = os.environ.copy()
    env["BUILD"] = rb.build
    env["SRC"] = rb.rust_root
    env["BOOTSTRAP_PARENT_ID"] = str(os.getpid())
    rb.run(args, env)

def main():
    start_time = time()
    help_triggered = ('-h' in sys.argv) or ('--help' in sys.argv) or (len(sys.argv) == 1)
    try:
        bootstrap()
        if not help_triggered:
            print("Build completed successfully in %s" % format_build_time(time() - start_time))
    except (SystemExit, KeyboardInterrupt) as e:
        if hasattr(e, 'code') and isinstance(e.code, int):
            exit_code = e.code
        else:
            exit_code = 1
            print(e)
        if not help_triggered:
            print("Build completed unsuccessfully in %s" % format_build_time(time() - start_time))
        sys.exit(exit_code)

if __name__ == '__main__':
    main()
