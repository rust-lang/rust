from __future__ import absolute_import, division, print_function
import argparse
import contextlib
import datetime
import distutils.version
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

from time import time, sleep

def support_xz():
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        with tarfile.open(temp_path, "w:xz"):
            pass
        return True
    except tarfile.CompressionError:
        return False

def get(base, url, path, checksums, verbose=False):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        if url not in checksums:
            raise RuntimeError(("src/stage0.json doesn't contain a checksum for {}. "
                                "Pre-built artifacts might not be available for this "
                                "target at this time, see https://doc.rust-lang.org/nightly"
                                "/rustc/platform-support.html for more information.")
                                .format(url))
        sha256 = checksums[url]
        if os.path.exists(path):
            if verify(path, sha256, False):
                if verbose:
                    print("using already-download file", path)
                return
            else:
                if verbose:
                    print("ignoring already-download file",
                        path, "due to failed verification")
                os.unlink(path)
        download(temp_path, "{}/{}".format(base, url), True, verbose)
        if not verify(temp_path, sha256, verbose):
            raise RuntimeError("failed verification")
        if verbose:
            print("moving {} to {}".format(temp_path, path))
        shutil.move(temp_path, path)
    finally:
        if os.path.isfile(temp_path):
            if verbose:
                print("removing", temp_path)
            os.unlink(temp_path)


def download(path, url, probably_big, verbose):
    for _ in range(0, 4):
        try:
            _download(path, url, probably_big, verbose, True)
            return
        except RuntimeError:
            print("\nspurious failure, trying again")
    _download(path, url, probably_big, verbose, False)


def _download(path, url, probably_big, verbose, exception):
    # Try to use curl (potentially available on win32
    #    https://devblogs.microsoft.com/commandline/tar-and-curl-come-to-windows/)
    # If an error occurs:
    #  - If we are on win32 fallback to powershell
    #  - Otherwise raise the error if appropriate
    if probably_big or verbose:
        print("downloading {}".format(url))

    platform_is_win32 = sys.platform == 'win32'
    try:
        if probably_big or verbose:
            option = "-#"
        else:
            option = "-s"
        # If curl is not present on Win32, we should not sys.exit
        #   but raise `CalledProcessError` or `OSError` instead
        require(["curl", "--version"], exception=platform_is_win32)
        run(["curl", option,
             "-L", # Follow redirect.
             "-y", "30", "-Y", "10",    # timeout if speed is < 10 bytes/sec for > 30 seconds
             "--connect-timeout", "30",  # timeout if cannot connect within 30 seconds
             "--retry", "3", "-Sf", "-o", path, url],
            verbose=verbose,
            exception=True, # Will raise RuntimeError on failure
        )
    except (subprocess.CalledProcessError, OSError, RuntimeError):
        # see http://serverfault.com/questions/301128/how-to-download
        if platform_is_win32:
            run(["PowerShell.exe", "/nologo", "-Command",
                 "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;",
                 "(New-Object System.Net.WebClient).DownloadFile('{}', '{}')".format(url, path)],
                verbose=verbose,
                exception=exception)
        # Check if the RuntimeError raised by run(curl) should be silenced
        elif verbose or exception:
            raise


def verify(path, expected, verbose):
    """Check if the sha256 sum of the given path is valid"""
    if verbose:
        print("verifying", path)
    with open(path, "rb") as source:
        found = hashlib.sha256(source.read()).hexdigest()
    verified = found == expected
    if not verified:
        print("invalid checksum:\n"
              "    found:    {}\n"
              "    expected: {}".format(found, expected))
    return verified


def unpack(tarball, tarball_suffix, dst, verbose=False, match=None):
    """Unpack the given tarball file"""
    print("extracting", tarball)
    fname = os.path.basename(tarball).replace(tarball_suffix, "")
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


def run(args, verbose=False, exception=False, is_bootstrap=False, **kwargs):
    """Run a child program in a new process"""
    if verbose:
        print("running: " + ' '.join(args))
    sys.stdout.flush()
    # Ensure that the .exe is used on Windows just in case a Linux ELF has been
    # compiled in the same directory.
    if os.name == 'nt' and not args[0].endswith('.exe'):
        args[0] += '.exe'
    # Use Popen here instead of call() as it apparently allows powershell on
    # Windows to not lock up waiting for input presumably.
    ret = subprocess.Popen(args, **kwargs)
    code = ret.wait()
    if code != 0:
        err = "failed to run: " + ' '.join(args)
        if verbose or exception:
            raise RuntimeError(err)
        # For most failures, we definitely do want to print this error, or the user will have no
        # idea what went wrong. But when we've successfully built bootstrap and it failed, it will
        # have already printed an error above, so there's no need to print the exact command we're
        # running.
        if is_bootstrap:
            sys.exit(1)
        else:
            sys.exit(err)


def require(cmd, exit=True, exception=False):
    '''Run a command, returning its output.
    On error,
        If `exception` is `True`, raise the error
        Otherwise If `exit` is `True`, exit the process
        Else return None.'''
    try:
        return subprocess.check_output(cmd).strip()
    except (subprocess.CalledProcessError, OSError) as exc:
        if exception:
            raise
        elif exit:
            print("error: unable to run `{}`: {}".format(' '.join(cmd), exc))
            print("Please make sure it's installed and in the path.")
            sys.exit(1)
        return None



def format_build_time(duration):
    """Return a nicer format for build time

    >>> format_build_time('300')
    '0:05:00'
    """
    return str(datetime.timedelta(seconds=int(duration)))


def default_build_triple(verbose):
    """Build triple as in LLVM"""
    # If the user already has a host build triple with an existing `rustc`
    # install, use their preference. This fixes most issues with Windows builds
    # being detected as GNU instead of MSVC.
    default_encoding = sys.getdefaultencoding()
    try:
        version = subprocess.check_output(["rustc", "--version", "--verbose"],
                stderr=subprocess.DEVNULL)
        version = version.decode(default_encoding)
        host = next(x for x in version.split('\n') if x.startswith("host: "))
        triple = host.split("host: ")[1]
        if verbose:
            print("detected default triple {} from pre-installed rustc".format(triple))
        return triple
    except Exception as e:
        if verbose:
            print("pre-installed rustc not detected: {}".format(e))
            print("falling back to auto-detect")

    required = sys.platform != 'win32'
    ostype = require(["uname", "-s"], exit=required)
    cputype = require(['uname', '-m'], exit=required)

    # If we do not have `uname`, assume Windows.
    if ostype is None or cputype is None:
        return 'x86_64-pc-windows-msvc'

    ostype = ostype.decode(default_encoding)
    cputype = cputype.decode(default_encoding)

    # The goal here is to come up with the same triple as LLVM would,
    # at least for the subset of platforms we're willing to target.
    ostype_mapper = {
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
        ostype = 'pc-solaris'
        # On Solaris, uname -m will return a machine classification instead
        # of a cpu type, so uname -p is recommended instead.  However, the
        # output from that option is too generic for our purposes (it will
        # always emit 'i386' on x86/amd64 systems).  As such, isainfo -k
        # must be used instead.
        cputype = require(['isainfo', '-k']).decode(default_encoding)
        # sparc cpus have sun as a target vendor
        if 'sparc' in cputype:
            ostype = 'sun-solaris'
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
    elif sys.platform == 'win32':
        # Some Windows platforms might have a `uname` command that returns a
        # non-standard string (e.g. gnuwin32 tools returns `windows32`). In
        # these cases, fall back to using sys.platform.
        return 'x86_64-pc-windows-msvc'
    else:
        err = "unknown OS type: {}".format(ostype)
        sys.exit(err)

    if cputype in ['powerpc', 'riscv'] and ostype == 'unknown-freebsd':
        cputype = subprocess.check_output(
              ['uname', '-p']).strip().decode(default_encoding)
    cputype_mapper = {
        'BePC': 'i686',
        'aarch64': 'aarch64',
        'amd64': 'x86_64',
        'arm64': 'aarch64',
        'i386': 'i686',
        'i486': 'i686',
        'i686': 'i686',
        'i786': 'i686',
        'm68k': 'm68k',
        'powerpc': 'powerpc',
        'powerpc64': 'powerpc64',
        'powerpc64le': 'powerpc64le',
        'ppc': 'powerpc',
        'ppc64': 'powerpc64',
        'ppc64le': 'powerpc64le',
        'riscv64': 'riscv64gc',
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
        elif ostype == 'unknown-freebsd':
            cputype = subprocess.check_output(
                ['uname', '-p']).strip().decode(default_encoding)
            ostype = 'unknown-freebsd'
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
    elif cputype == 'sparc' or cputype == 'sparcv9' or cputype == 'sparc64':
        pass
    else:
        err = "unknown cpu type: {}".format(cputype)
        sys.exit(err)

    return "{}-{}".format(cputype, ostype)


@contextlib.contextmanager
def output(filepath):
    tmp = filepath + '.tmp'
    with open(tmp, 'w') as f:
        yield f
    try:
        if os.path.exists(filepath):
            os.remove(filepath)  # PermissionError/OSError on Win32 if in use
    except OSError:
        shutil.copy2(tmp, filepath)
        os.remove(tmp)
        return
    os.rename(tmp, filepath)


class Stage0Toolchain:
    def __init__(self, stage0_payload):
        self.date = stage0_payload["date"]
        self.version = stage0_payload["version"]

    def channel(self):
        return self.version + "-" + self.date


class RustBuild(object):
    """Provide all the methods required to build Rust"""
    def __init__(self):
        self.checksums_sha256 = {}
        self.stage0_compiler = None
        self._download_url = ''
        self.build = ''
        self.build_dir = ''
        self.clean = False
        self.config_toml = ''
        self.rust_root = ''
        self.use_locked_deps = ''
        self.use_vendored_sources = ''
        self.verbose = False
        self.git_version = None
        self.nix_deps_dir = None

    def download_toolchain(self):
        """Fetch the build system for Rust, written in Rust

        This method will build a cache directory, then it will fetch the
        tarball which has the stage0 compiler used to then bootstrap the Rust
        compiler itself.

        Each downloaded tarball is extracted, after that, the script
        will move all the content to the right place.
        """
        rustc_channel = self.stage0_compiler.version
        bin_root = self.bin_root()

        key = self.stage0_compiler.date
        if self.rustc().startswith(bin_root) and \
                (not os.path.exists(self.rustc()) or
                 self.program_out_of_date(self.rustc_stamp(), key)):
            if os.path.exists(bin_root):
                shutil.rmtree(bin_root)
            tarball_suffix = '.tar.xz' if support_xz() else '.tar.gz'
            filename = "rust-std-{}-{}{}".format(
                rustc_channel, self.build, tarball_suffix)
            pattern = "rust-std-{}".format(self.build)
            self._download_component_helper(filename, pattern, tarball_suffix)
            filename = "rustc-{}-{}{}".format(rustc_channel, self.build,
                                              tarball_suffix)
            self._download_component_helper(filename, "rustc", tarball_suffix)
            filename = "cargo-{}-{}{}".format(rustc_channel, self.build,
                                            tarball_suffix)
            self._download_component_helper(filename, "cargo", tarball_suffix)
            self.fix_bin_or_dylib("{}/bin/cargo".format(bin_root))

            self.fix_bin_or_dylib("{}/bin/rustc".format(bin_root))
            self.fix_bin_or_dylib("{}/bin/rustdoc".format(bin_root))
            lib_dir = "{}/lib".format(bin_root)
            for lib in os.listdir(lib_dir):
                if lib.endswith(".so"):
                    self.fix_bin_or_dylib(os.path.join(lib_dir, lib))
            with output(self.rustc_stamp()) as rust_stamp:
                rust_stamp.write(key)

    def _download_component_helper(
        self, filename, pattern, tarball_suffix,
    ):
        key = self.stage0_compiler.date
        cache_dst = os.path.join(self.build_dir, "cache")
        rustc_cache = os.path.join(cache_dst, key)
        if not os.path.exists(rustc_cache):
            os.makedirs(rustc_cache)

        base = self._download_url
        url = "dist/{}".format(key)
        tarball = os.path.join(rustc_cache, filename)
        if not os.path.exists(tarball):
            get(
                base,
                "{}/{}".format(url, filename),
                tarball,
                self.checksums_sha256,
                verbose=self.verbose,
            )
        unpack(tarball, tarball_suffix, self.bin_root(), match=pattern, verbose=self.verbose)

    def fix_bin_or_dylib(self, fname):
        """Modifies the interpreter section of 'fname' to fix the dynamic linker,
        or the RPATH section, to fix the dynamic library search path

        This method is only required on NixOS and uses the PatchELF utility to
        change the interpreter/RPATH of ELF executables.

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

        # If the user has asked binaries to be patched for Nix, then
        # don't check for NixOS or `/lib`, just continue to the patching.
        if self.get_toml('patch-binaries-for-nix', 'build') != 'true':
            # Use `/etc/os-release` instead of `/etc/NIXOS`.
            # The latter one does not exist on NixOS when using tmpfs as root.
            try:
                with open("/etc/os-release", "r") as f:
                    if not any(l.strip() in ["ID=nixos", "ID='nixos'", 'ID="nixos"'] for l in f):
                        return
            except FileNotFoundError:
                return
            if os.path.exists("/lib"):
                return

        # At this point we're pretty sure the user is running NixOS or
        # using Nix
        nix_os_msg = "info: you seem to be using Nix. Attempting to patch"
        print(nix_os_msg, fname)

        # Only build `.nix-deps` once.
        nix_deps_dir = self.nix_deps_dir
        if not nix_deps_dir:
            # Run `nix-build` to "build" each dependency (which will likely reuse
            # the existing `/nix/store` copy, or at most download a pre-built copy).
            #
            # Importantly, we create a gc-root called `.nix-deps` in the `build/`
            # directory, but still reference the actual `/nix/store` path in the rpath
            # as it makes it significantly more robust against changes to the location of
            # the `.nix-deps` location.
            #
            # bintools: Needed for the path of `ld-linux.so` (via `nix-support/dynamic-linker`).
            # zlib: Needed as a system dependency of `libLLVM-*.so`.
            # patchelf: Needed for patching ELF binaries (see doc comment above).
            nix_deps_dir = "{}/{}".format(self.build_dir, ".nix-deps")
            nix_expr = '''
            with (import <nixpkgs> {});
            symlinkJoin {
              name = "rust-stage0-dependencies";
              paths = [
                zlib
                patchelf
                stdenv.cc.bintools
              ];
            }
            '''
            try:
                subprocess.check_output([
                    "nix-build", "-E", nix_expr, "-o", nix_deps_dir,
                ])
            except subprocess.CalledProcessError as reason:
                print("warning: failed to call nix-build:", reason)
                return
            self.nix_deps_dir = nix_deps_dir

        patchelf = "{}/bin/patchelf".format(nix_deps_dir)
        rpath_entries = [
            # Relative default, all binary and dynamic libraries we ship
            # appear to have this (even when `../lib` is redundant).
            "$ORIGIN/../lib",
            os.path.join(os.path.realpath(nix_deps_dir), "lib")
        ]
        patchelf_args = ["--set-rpath", ":".join(rpath_entries)]
        if not fname.endswith(".so"):
            # Finally, set the corret .interp for binaries
            with open("{}/nix-support/dynamic-linker".format(nix_deps_dir)) as dynamic_linker:
                patchelf_args += ["--set-interpreter", dynamic_linker.read().rstrip()]

        try:
            subprocess.check_output([patchelf] + patchelf_args + [fname])
        except subprocess.CalledProcessError as reason:
            print("warning: failed to call patchelf:", reason)
            return

    def rustc_stamp(self):
        """Return the path for .rustc-stamp at the given stage

        >>> rb = RustBuild()
        >>> rb.build_dir = "build"
        >>> rb.rustc_stamp() == os.path.join("build", "stage0", ".rustc-stamp")
        True
        """
        return os.path.join(self.bin_root(), '.rustc-stamp')

    def program_out_of_date(self, stamp_path, key):
        """Check if the given program stamp is out of date"""
        if not os.path.exists(stamp_path) or self.clean:
            return True
        with open(stamp_path, 'r') as stamp:
            return key != stamp.read()

    def bin_root(self):
        """Return the binary root directory for the given stage

        >>> rb = RustBuild()
        >>> rb.build_dir = "build"
        >>> rb.bin_root() == os.path.join("build", "stage0")
        True

        When the 'build' property is given should be a nested directory:

        >>> rb.build = "devel"
        >>> rb.bin_root() == os.path.join("build", "devel", "stage0")
        True
        """
        subdir = "stage0"
        return os.path.join(self.build_dir, self.build, subdir)

    def get_toml(self, key, section=None):
        """Returns the value of the given key in config.toml, otherwise returns None

        >>> rb = RustBuild()
        >>> rb.config_toml = 'key1 = "value1"\\nkey2 = "value2"'
        >>> rb.get_toml("key2")
        'value2'

        If the key does not exist, the result is None:

        >>> rb.get_toml("key3") is None
        True

        Optionally also matches the section the key appears in

        >>> rb.config_toml = '[a]\\nkey = "value1"\\n[b]\\nkey = "value2"'
        >>> rb.get_toml('key', 'a')
        'value1'
        >>> rb.get_toml('key', 'b')
        'value2'
        >>> rb.get_toml('key', 'c') is None
        True

        >>> rb.config_toml = 'key1 = true'
        >>> rb.get_toml("key1")
        'true'
        """

        cur_section = None
        for line in self.config_toml.splitlines():
            section_match = re.match(r'^\s*\[(.*)\]\s*$', line)
            if section_match is not None:
                cur_section = section_match.group(1)

            match = re.match(r'^{}\s*=(.*)$'.format(key), line)
            if match is not None:
                value = match.group(1)
                if section is None or section == cur_section:
                    return self.get_string(value) or value.strip()
        return None

    def cargo(self):
        """Return config path for cargo"""
        return self.program_config('cargo')

    def rustc(self):
        """Return config path for rustc"""
        return self.program_config('rustc')

    def program_config(self, program):
        """Return config path for the given program at the given stage

        >>> rb = RustBuild()
        >>> rb.config_toml = 'rustc = "rustc"\\n'
        >>> rb.program_config('rustc')
        'rustc'
        >>> rb.config_toml = ''
        >>> cargo_path = rb.program_config('cargo')
        >>> cargo_path.rstrip(".exe") == os.path.join(rb.bin_root(),
        ... "bin", "cargo")
        True
        """
        config = self.get_toml(program)
        if config:
            return os.path.expanduser(config)
        return os.path.join(self.bin_root(), "bin", "{}{}".format(
            program, self.exe_suffix()))

    @staticmethod
    def get_string(line):
        """Return the value between double quotes

        >>> RustBuild.get_string('    "devel"   ')
        'devel'
        >>> RustBuild.get_string("    'devel'   ")
        'devel'
        >>> RustBuild.get_string('devel') is None
        True
        >>> RustBuild.get_string('    "devel   ')
        ''
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

    def bootstrap_binary(self):
        """Return the path of the bootstrap binary

        >>> rb = RustBuild()
        >>> rb.build_dir = "build"
        >>> rb.bootstrap_binary() == os.path.join("build", "bootstrap",
        ... "debug", "bootstrap")
        True
        """
        return os.path.join(self.build_dir, "bootstrap", "debug", "bootstrap")

    def build_bootstrap(self, color):
        """Build bootstrap"""
        print("Building rustbuild")
        build_dir = os.path.join(self.build_dir, "bootstrap")
        if self.clean and os.path.exists(build_dir):
            shutil.rmtree(build_dir)
        env = os.environ.copy()
        # `CARGO_BUILD_TARGET` breaks bootstrap build.
        # See also: <https://github.com/rust-lang/rust/issues/70208>.
        if "CARGO_BUILD_TARGET" in env:
            del env["CARGO_BUILD_TARGET"]
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

        # preserve existing RUSTFLAGS
        env.setdefault("RUSTFLAGS", "")
        build_section = "target.{}".format(self.build)
        target_features = []
        if self.get_toml("crt-static", build_section) == "true":
            target_features += ["+crt-static"]
        elif self.get_toml("crt-static", build_section) == "false":
            target_features += ["-crt-static"]
        if target_features:
            env["RUSTFLAGS"] += " -C target-feature=" + (",".join(target_features))
        target_linker = self.get_toml("linker", build_section)
        if target_linker is not None:
            env["RUSTFLAGS"] += " -C linker=" + target_linker
        env["RUSTFLAGS"] += " -Wrust_2018_idioms -Wunused_lifetimes"
        env["RUSTFLAGS"] += " -Wsemicolon_in_expressions_from_macros"
        if self.get_toml("deny-warnings", "rust") != "false":
            env["RUSTFLAGS"] += " -Dwarnings"

        env["PATH"] = os.path.join(self.bin_root(), "bin") + \
            os.pathsep + env["PATH"]
        if not os.path.isfile(self.cargo()):
            raise Exception("no cargo executable found at `{}`".format(
                self.cargo()))
        args = [self.cargo(), "build", "--manifest-path",
                os.path.join(self.rust_root, "src/bootstrap/Cargo.toml")]
        for _ in range(0, self.verbose):
            args.append("--verbose")
        if self.use_locked_deps:
            args.append("--locked")
        if self.use_vendored_sources:
            args.append("--frozen")
        if self.get_toml("metrics", "build"):
            args.append("--features")
            args.append("build-metrics")
        if color == "always":
            args.append("--color=always")
        elif color == "never":
            args.append("--color=never")

        run(args, env=env, verbose=self.verbose)

    def build_triple(self):
        """Build triple as in LLVM

        Note that `default_build_triple` is moderately expensive,
        so use `self.build` where possible.
        """
        config = self.get_toml('build')
        if config:
            return config
        return default_build_triple(self.verbose)

    def set_dist_environment(self, url):
        """Set download URL for normal environment"""
        if 'RUSTUP_DIST_SERVER' in os.environ:
            self._download_url = os.environ['RUSTUP_DIST_SERVER']
        else:
            self._download_url = url

    def check_vendored_status(self):
        """Check that vendoring is configured properly"""
        # keep this consistent with the equivalent check in rustbuild:
        # https://github.com/rust-lang/rust/blob/a8a33cf27166d3eabaffc58ed3799e054af3b0c6/src/bootstrap/lib.rs#L399-L405
        if 'SUDO_USER' in os.environ and not self.use_vendored_sources:
            if os.getuid() == 0:
                self.use_vendored_sources = True
                print('info: looks like you\'re trying to run this command as root')
                print('      and so in order to preserve your $HOME this will now')
                print('      use vendored sources by default.')

        cargo_dir = os.path.join(self.rust_root, '.cargo')
        if self.use_vendored_sources:
            vendor_dir = os.path.join(self.rust_root, 'vendor')
            if not os.path.exists(vendor_dir):
                sync_dirs = "--sync ./src/tools/rust-analyzer/Cargo.toml " \
                            "--sync ./compiler/rustc_codegen_cranelift/Cargo.toml " \
                            "--sync ./src/bootstrap/Cargo.toml "
                print('error: vendoring required, but vendor directory does not exist.')
                print('       Run `cargo vendor {}` to initialize the '
                      'vendor directory.'.format(sync_dirs))
                print('Alternatively, use the pre-vendored `rustc-src` dist component.')
                raise Exception("{} not found".format(vendor_dir))

            if not os.path.exists(cargo_dir):
                print('error: vendoring required, but .cargo/config does not exist.')
                raise Exception("{} not found".format(cargo_dir))
        else:
            if os.path.exists(cargo_dir):
                shutil.rmtree(cargo_dir)

def bootstrap(help_triggered):
    """Configure, fetch, build and run the initial bootstrap"""

    # If the user is asking for help, let them know that the whole download-and-build
    # process has to happen before anything is printed out.
    if help_triggered:
        print("info: Downloading and building bootstrap before processing --help")
        print("      command. See src/bootstrap/README.md for help with common")
        print("      commands.")

    parser = argparse.ArgumentParser(description='Build rust')
    parser.add_argument('--config')
    parser.add_argument('--build-dir')
    parser.add_argument('--build')
    parser.add_argument('--color', choices=['always', 'never', 'auto'])
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('-v', '--verbose', action='count', default=0)

    args = [a for a in sys.argv if a != '-h' and a != '--help']
    args, _ = parser.parse_known_args(args)

    # Configure initial bootstrap
    build = RustBuild()
    build.rust_root = os.path.abspath(os.path.join(__file__, '../../..'))
    build.verbose = args.verbose
    build.clean = args.clean

    # Read from `--config`, then `RUST_BOOTSTRAP_CONFIG`, then `./config.toml`,
    # then `config.toml` in the root directory.
    toml_path = args.config or os.getenv('RUST_BOOTSTRAP_CONFIG')
    using_default_path = toml_path is None
    if using_default_path:
        toml_path = 'config.toml'
        if not os.path.exists(toml_path):
            toml_path = os.path.join(build.rust_root, toml_path)

    # Give a hard error if `--config` or `RUST_BOOTSTRAP_CONFIG` are set to a missing path,
    # but not if `config.toml` hasn't been created.
    if not using_default_path or os.path.exists(toml_path):
        with open(toml_path) as config:
            build.config_toml = config.read()

    profile = build.get_toml('profile')
    if profile is not None:
        include_file = 'config.{}.toml'.format(profile)
        include_dir = os.path.join(build.rust_root, 'src', 'bootstrap', 'defaults')
        include_path = os.path.join(include_dir, include_file)
        # HACK: This works because `build.get_toml()` returns the first match it finds for a
        # specific key, so appending our defaults at the end allows the user to override them
        with open(include_path) as included_toml:
            build.config_toml += os.linesep + included_toml.read()

    config_verbose = build.get_toml('verbose', 'build')
    if config_verbose is not None:
        build.verbose = max(build.verbose, int(config_verbose))

    build.use_vendored_sources = build.get_toml('vendor', 'build') == 'true'

    build.use_locked_deps = build.get_toml('locked-deps', 'build') == 'true'

    build.check_vendored_status()

    build_dir = args.build_dir or build.get_toml('build-dir', 'build') or 'build'
    build.build_dir = os.path.abspath(build_dir)

    with open(os.path.join(build.rust_root, "src", "stage0.json")) as f:
        data = json.load(f)
    build.checksums_sha256 = data["checksums_sha256"]
    build.stage0_compiler = Stage0Toolchain(data["compiler"])

    build.set_dist_environment(data["config"]["dist_server"])

    build.build = args.build or build.build_triple()

    if not os.path.exists(build.build_dir):
        os.makedirs(build.build_dir)

    # Fetch/build the bootstrap
    build.download_toolchain()
    sys.stdout.flush()
    build.build_bootstrap(args.color)
    sys.stdout.flush()

    # Run the bootstrap
    args = [build.bootstrap_binary()]
    args.extend(sys.argv[1:])
    env = os.environ.copy()
    env["BOOTSTRAP_PARENT_ID"] = str(os.getpid())
    env["BOOTSTRAP_PYTHON"] = sys.executable
    run(args, env=env, verbose=build.verbose, is_bootstrap=True)


def main():
    """Entry point for the bootstrap process"""
    start_time = time()

    # x.py help <cmd> ...
    if len(sys.argv) > 1 and sys.argv[1] == 'help':
        sys.argv = [sys.argv[0], '-h'] + sys.argv[2:]

    help_triggered = (
        '-h' in sys.argv) or ('--help' in sys.argv) or (len(sys.argv) == 1)
    try:
        bootstrap(help_triggered)
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
