from __future__ import absolute_import, division, print_function
import argparse
import contextlib
import datetime
import distutils.version
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

from time import time

def support_xz():
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        with tarfile.open(temp_path, "w:xz"):
            pass
        return True
    except tarfile.CompressionError:
        return False

def get(url, path, verbose=False, do_verify=True):
    suffix = '.sha256'
    sha_url = url + suffix
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as sha_file:
        sha_path = sha_file.name

    try:
        if do_verify:
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
        if do_verify and not verify(temp_path, sha_path, verbose):
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
             "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;",
             "(New-Object System.Net.WebClient).DownloadFile('{}', '{}')".format(url, path)],
            verbose=verbose,
            exception=exception)
    else:
        if probably_big or verbose:
            option = "-#"
        else:
            option = "-s"
        require(["curl", "--version"])
        run(["curl", option,
             "-y", "30", "-Y", "10",    # timeout if speed is < 10 bytes/sec for > 30 seconds
             "--connect-timeout", "30",  # timeout if cannot connect within 30 seconds
             "--retry", "3", "-Sf", "-o", path, url],
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


def require(cmd, exit=True):
    '''Run a command, returning its output.
    On error,
        If `exit` is `True`, exit the process.
        Otherwise, return None.'''
    try:
        return subprocess.check_output(cmd).strip()
    except (subprocess.CalledProcessError, OSError) as exc:
        if not exit:
            return None
        print("error: unable to run `{}`: {}".format(' '.join(cmd), exc))
        print("Please make sure it's installed and in the path.")
        sys.exit(1)


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
        ostype = 'sun-solaris'
        # On Solaris, uname -m will return a machine classification instead
        # of a cpu type, so uname -p is recommended instead.  However, the
        # output from that option is too generic for our purposes (it will
        # always emit 'i386' on x86/amd64 systems).  As such, isainfo -k
        # must be used instead.
        cputype = require(['isainfo', '-k']).decode(default_encoding)
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

    if cputype == 'powerpc' and ostype == 'unknown-freebsd':
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
        os.remove(filepath)  # PermissionError/OSError on Win32 if in use
        os.rename(tmp, filepath)
    except OSError:
        shutil.copy2(tmp, filepath)
        os.remove(tmp)


class RustBuild(object):
    """Provide all the methods required to build Rust"""
    def __init__(self):
        self.cargo_channel = ''
        self.date = ''
        self._download_url = ''
        self.rustc_channel = ''
        self.rustfmt_channel = ''
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
        rustfmt_channel = self.rustfmt_channel

        if self.rustc().startswith(self.bin_root()) and \
                (not os.path.exists(self.rustc()) or
                 self.program_out_of_date(self.rustc_stamp())):
            if os.path.exists(self.bin_root()):
                shutil.rmtree(self.bin_root())
            tarball_suffix = '.tar.xz' if support_xz() else '.tar.gz'
            filename = "rust-std-{}-{}{}".format(
                rustc_channel, self.build, tarball_suffix)
            pattern = "rust-std-{}".format(self.build)
            self._download_stage0_helper(filename, pattern, tarball_suffix)

            filename = "rustc-{}-{}{}".format(rustc_channel, self.build,
                                              tarball_suffix)
            self._download_stage0_helper(filename, "rustc", tarball_suffix)
            self.fix_bin_or_dylib("{}/bin/rustc".format(self.bin_root()))
            self.fix_bin_or_dylib("{}/bin/rustdoc".format(self.bin_root()))
            lib_dir = "{}/lib".format(self.bin_root())
            for lib in os.listdir(lib_dir):
                if lib.endswith(".so"):
                    self.fix_bin_or_dylib("{}/{}".format(lib_dir, lib))
            with output(self.rustc_stamp()) as rust_stamp:
                rust_stamp.write(self.date)

        if self.cargo().startswith(self.bin_root()) and \
                (not os.path.exists(self.cargo()) or
                 self.program_out_of_date(self.cargo_stamp())):
            tarball_suffix = '.tar.xz' if support_xz() else '.tar.gz'
            filename = "cargo-{}-{}{}".format(cargo_channel, self.build,
                                              tarball_suffix)
            self._download_stage0_helper(filename, "cargo", tarball_suffix)
            self.fix_bin_or_dylib("{}/bin/cargo".format(self.bin_root()))
            with output(self.cargo_stamp()) as cargo_stamp:
                cargo_stamp.write(self.date)

        if self.rustfmt() and self.rustfmt().startswith(self.bin_root()) and (
            not os.path.exists(self.rustfmt())
            or self.program_out_of_date(self.rustfmt_stamp(), self.rustfmt_channel)
        ):
            if rustfmt_channel:
                tarball_suffix = '.tar.xz' if support_xz() else '.tar.gz'
                [channel, date] = rustfmt_channel.split('-', 1)
                filename = "rustfmt-{}-{}{}".format(channel, self.build, tarball_suffix)
                self._download_stage0_helper(filename, "rustfmt-preview", tarball_suffix, date)
                self.fix_bin_or_dylib("{}/bin/rustfmt".format(self.bin_root()))
                self.fix_bin_or_dylib("{}/bin/cargo-fmt".format(self.bin_root()))
                with output(self.rustfmt_stamp()) as rustfmt_stamp:
                    rustfmt_stamp.write(self.date + self.rustfmt_channel)

        if self.downloading_llvm():
            # We want the most recent LLVM submodule update to avoid downloading
            # LLVM more often than necessary.
            #
            # This git command finds that commit SHA, looking for bors-authored
            # merges that modified src/llvm-project.
            #
            # This works even in a repository that has not yet initialized
            # submodules.
            llvm_sha = subprocess.check_output([
                "git", "log", "--author=bors", "--format=%H", "-n1",
                "-m", "--first-parent",
                "--",
                "src/llvm-project",
                "src/bootstrap/download-ci-llvm-stamp",
            ]).decode(sys.getdefaultencoding()).strip()
            llvm_assertions = self.get_toml('assertions', 'llvm') == 'true'
            if self.program_out_of_date(self.llvm_stamp(), llvm_sha + str(llvm_assertions)):
                self._download_ci_llvm(llvm_sha, llvm_assertions)
                for binary in ["llvm-config", "FileCheck"]:
                    self.fix_bin_or_dylib("{}/bin/{}".format(self.llvm_root(), binary))
                with output(self.llvm_stamp()) as llvm_stamp:
                    llvm_stamp.write(self.date + llvm_sha + str(llvm_assertions))

    def downloading_llvm(self):
        opt = self.get_toml('download-ci-llvm', 'llvm')
        return opt == "true" \
            or (opt == "if-available" and self.build == "x86_64-unknown-linux-gnu")

    def _download_stage0_helper(self, filename, pattern, tarball_suffix, date=None):
        if date is None:
            date = self.date
        cache_dst = os.path.join(self.build_dir, "cache")
        rustc_cache = os.path.join(cache_dst, date)
        if not os.path.exists(rustc_cache):
            os.makedirs(rustc_cache)

        url = "{}/dist/{}".format(self._download_url, date)
        tarball = os.path.join(rustc_cache, filename)
        if not os.path.exists(tarball):
            get("{}/{}".format(url, filename), tarball, verbose=self.verbose)
        unpack(tarball, tarball_suffix, self.bin_root(), match=pattern, verbose=self.verbose)

    def _download_ci_llvm(self, llvm_sha, llvm_assertions):
        cache_prefix = "llvm-{}-{}".format(llvm_sha, llvm_assertions)
        cache_dst = os.path.join(self.build_dir, "cache")
        rustc_cache = os.path.join(cache_dst, cache_prefix)
        if not os.path.exists(rustc_cache):
            os.makedirs(rustc_cache)

        url = "https://ci-artifacts.rust-lang.org/rustc-builds/{}".format(llvm_sha)
        if llvm_assertions:
            url = url.replace('rustc-builds', 'rustc-builds-alt')
        tarball_suffix = '.tar.xz' if support_xz() else '.tar.gz'
        filename = "rust-dev-nightly-" + self.build + tarball_suffix
        tarball = os.path.join(rustc_cache, filename)
        if not os.path.exists(tarball):
            get("{}/{}".format(url, filename), tarball, verbose=self.verbose, do_verify=False)
        unpack(tarball, tarball_suffix, self.llvm_root(),
                match="rust-dev",
                verbose=self.verbose)

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

        if not os.path.exists("/etc/NIXOS"):
            return
        if os.path.exists("/lib"):
            return

        # At this point we're pretty sure the user is running NixOS
        nix_os_msg = "info: you seem to be running NixOS. Attempting to patch"
        print(nix_os_msg, fname)

        # Only build `stage0/.nix-deps` once.
        nix_deps_dir = self.nix_deps_dir
        if not nix_deps_dir:
            nix_deps_dir = "{}/.nix-deps".format(self.bin_root())
            if not os.path.exists(nix_deps_dir):
                os.makedirs(nix_deps_dir)

            nix_deps = [
                # Needed for the path of `ld-linux.so` (via `nix-support/dynamic-linker`).
                "stdenv.cc.bintools",

                # Needed as a system dependency of `libLLVM-*.so`.
                "zlib",

                # Needed for patching ELF binaries (see doc comment above).
                "patchelf",
            ]

            # Run `nix-build` to "build" each dependency (which will likely reuse
            # the existing `/nix/store` copy, or at most download a pre-built copy).
            # Importantly, we don't rely on `nix-build` printing the `/nix/store`
            # path on stdout, but use `-o` to symlink it into `stage0/.nix-deps/$dep`,
            # ensuring garbage collection will never remove the `/nix/store` path
            # (which would break our patched binaries that hardcode those paths).
            for dep in nix_deps:
                try:
                    subprocess.check_output([
                        "nix-build", "<nixpkgs>",
                        "-A", dep,
                        "-o", "{}/{}".format(nix_deps_dir, dep),
                    ])
                except subprocess.CalledProcessError as reason:
                    print("warning: failed to call nix-build:", reason)
                    return

            self.nix_deps_dir = nix_deps_dir

        patchelf = "{}/patchelf/bin/patchelf".format(nix_deps_dir)

        if fname.endswith(".so"):
            # Dynamic library, patch RPATH to point to system dependencies.
            dylib_deps = ["zlib"]
            rpath_entries = [
                # Relative default, all binary and dynamic libraries we ship
                # appear to have this (even when `../lib` is redundant).
                "$ORIGIN/../lib",
            ] + ["{}/{}/lib".format(nix_deps_dir, dep) for dep in dylib_deps]
            patchelf_args = ["--set-rpath", ":".join(rpath_entries)]
        else:
            bintools_dir = "{}/stdenv.cc.bintools".format(nix_deps_dir)
            with open("{}/nix-support/dynamic-linker".format(bintools_dir)) as dynamic_linker:
                patchelf_args = ["--set-interpreter", dynamic_linker.read().rstrip()]

        try:
            subprocess.check_output([patchelf] + patchelf_args + [fname])
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

    def rustfmt_stamp(self):
        """Return the path for .rustfmt-stamp

        >>> rb = RustBuild()
        >>> rb.build_dir = "build"
        >>> rb.rustfmt_stamp() == os.path.join("build", "stage0", ".rustfmt-stamp")
        True
        """
        return os.path.join(self.bin_root(), '.rustfmt-stamp')

    def llvm_stamp(self):
        """Return the path for .rustfmt-stamp

        >>> rb = RustBuild()
        >>> rb.build_dir = "build"
        >>> rb.llvm_stamp() == os.path.join("build", "ci-llvm", ".llvm-stamp")
        True
        """
        return os.path.join(self.llvm_root(), '.llvm-stamp')


    def program_out_of_date(self, stamp_path, extra=""):
        """Check if the given program stamp is out of date"""
        if not os.path.exists(stamp_path) or self.clean:
            return True
        with open(stamp_path, 'r') as stamp:
            return (self.date + extra) != stamp.read()

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

    def llvm_root(self):
        """Return the CI LLVM root directory

        >>> rb = RustBuild()
        >>> rb.build_dir = "build"
        >>> rb.llvm_root() == os.path.join("build", "ci-llvm")
        True

        When the 'build' property is given should be a nested directory:

        >>> rb.build = "devel"
        >>> rb.llvm_root() == os.path.join("build", "devel", "ci-llvm")
        True
        """
        return os.path.join(self.build_dir, self.build, "ci-llvm")

    def get_toml(self, key, section=None):
        """Returns the value of the given key in config.toml, otherwise returns None

        >>> rb = RustBuild()
        >>> rb.config_toml = 'key1 = "value1"\\nkey2 = "value2"'
        >>> rb.get_toml("key2")
        'value2'

        If the key does not exists, the result is None:

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

    def rustfmt(self):
        """Return config path for rustfmt"""
        if not self.rustfmt_channel:
            return None
        return self.program_config('rustfmt')

    def program_config(self, program):
        """Return config path for the given program

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

    def build_bootstrap(self):
        """Build bootstrap"""
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
        env["RUSTFLAGS"] += " -Cdebuginfo=2"

        build_section = "target.{}".format(self.build_triple())
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
        if self.get_toml("deny-warnings", "rust") != "false":
            env["RUSTFLAGS"] += " -Dwarnings"

        env["PATH"] = os.path.join(self.bin_root(), "bin") + \
            os.pathsep + env["PATH"]
        if not os.path.isfile(self.cargo()):
            raise Exception("no cargo executable found at `{}`".format(
                self.cargo()))
        args = [self.cargo(), "build", "--manifest-path",
                os.path.join(self.rust_root, "src/bootstrap/Cargo.toml")]
        for _ in range(1, self.verbose):
            args.append("--verbose")
        if self.use_locked_deps:
            args.append("--locked")
        if self.use_vendored_sources:
            args.append("--frozen")
        run(args, env=env, verbose=self.verbose)

    def build_triple(self):
        """Build triple as in LLVM"""
        config = self.get_toml('build')
        if config:
            return config
        return default_build_triple()

    def check_submodule(self, module, slow_submodules):
        if not slow_submodules:
            checked_out = subprocess.Popen(["git", "rev-parse", "HEAD"],
                                           cwd=os.path.join(self.rust_root, module),
                                           stdout=subprocess.PIPE)
            return checked_out
        else:
            return None

    def update_submodule(self, module, checked_out, recorded_submodules):
        module_path = os.path.join(self.rust_root, module)

        if checked_out is not None:
            default_encoding = sys.getdefaultencoding()
            checked_out = checked_out.communicate()[0].decode(default_encoding).strip()
            if recorded_submodules[module] == checked_out:
                return

        print("Updating submodule", module)

        run(["git", "submodule", "-q", "sync", module],
            cwd=self.rust_root, verbose=self.verbose)

        update_args = ["git", "submodule", "update", "--init", "--recursive"]
        if self.git_version >= distutils.version.LooseVersion("2.11.0"):
            update_args.append("--progress")
        update_args.append(module)
        run(update_args, cwd=self.rust_root, verbose=self.verbose, exception=True)

        run(["git", "reset", "-q", "--hard"],
            cwd=module_path, verbose=self.verbose)
        run(["git", "clean", "-qdfx"],
            cwd=module_path, verbose=self.verbose)

    def update_submodules(self):
        """Update submodules"""
        if (not os.path.exists(os.path.join(self.rust_root, ".git"))) or \
                self.get_toml('submodules') == "false":
            return

        default_encoding = sys.getdefaultencoding()

        # check the existence and version of 'git' command
        git_version_str = require(['git', '--version']).split()[2].decode(default_encoding)
        self.git_version = distutils.version.LooseVersion(git_version_str)

        slow_submodules = self.get_toml('fast-submodules') == "false"
        start_time = time()
        if slow_submodules:
            print('Unconditionally updating all submodules')
        else:
            print('Updating only changed submodules')
        default_encoding = sys.getdefaultencoding()
        submodules = [s.split(' ', 1)[1] for s in subprocess.check_output(
            ["git", "config", "--file",
             os.path.join(self.rust_root, ".gitmodules"),
             "--get-regexp", "path"]
        ).decode(default_encoding).splitlines()]
        filtered_submodules = []
        submodules_names = []
        llvm_checked_out = os.path.exists(os.path.join(self.rust_root, "src/llvm-project/.git"))
        for module in submodules:
            if module.endswith("llvm-project"):
                # Don't sync the llvm-project submodule either if an external LLVM
                # was provided, or if we are downloading LLVM. Also, if the
                # submodule has been initialized already, sync it anyways so that
                # it doesn't mess up contributor pull requests.
                if self.get_toml('llvm-config') or self.downloading_llvm():
                    if self.get_toml('lld') != 'true' and not llvm_checked_out:
                        continue
            check = self.check_submodule(module, slow_submodules)
            filtered_submodules.append((module, check))
            submodules_names.append(module)
        recorded = subprocess.Popen(["git", "ls-tree", "HEAD"] + submodules_names,
                                    cwd=self.rust_root, stdout=subprocess.PIPE)
        recorded = recorded.communicate()[0].decode(default_encoding).strip().splitlines()
        recorded_submodules = {}
        for data in recorded:
            data = data.split()
            recorded_submodules[data[3]] = data[2]
        for module in filtered_submodules:
            self.update_submodule(module[0], module[1], recorded_submodules)
        print("Submodules updated in %.2f seconds" % (time() - start_time))

    def set_normal_environment(self):
        """Set download URL for normal environment"""
        if 'RUSTUP_DIST_SERVER' in os.environ:
            self._download_url = os.environ['RUSTUP_DIST_SERVER']
        else:
            self._download_url = 'https://static.rust-lang.org'

    def set_dev_environment(self):
        """Set download URL for development environment"""
        if 'RUSTUP_DEV_DIST_SERVER' in os.environ:
            self._download_url = os.environ['RUSTUP_DEV_DIST_SERVER']
        else:
            self._download_url = 'https://dev-static.rust-lang.org'

    def check_vendored_status(self):
        """Check that vendoring is configured properly"""
        vendor_dir = os.path.join(self.rust_root, 'vendor')
        if 'SUDO_USER' in os.environ and not self.use_vendored_sources:
            if os.environ.get('USER') != os.environ['SUDO_USER']:
                self.use_vendored_sources = True
                print('info: looks like you are running this command under `sudo`')
                print('      and so in order to preserve your $HOME this will now')
                print('      use vendored sources by default.')
                if not os.path.exists(vendor_dir):
                    print('error: vendoring required, but vendor directory does not exist.')
                    print('       Run `cargo vendor` without sudo to initialize the '
                          'vendor directory.')
                    raise Exception("{} not found".format(vendor_dir))

        if self.use_vendored_sources:
            if not os.path.exists('.cargo'):
                os.makedirs('.cargo')
            with output('.cargo/config') as cargo_config:
                cargo_config.write(
                    "[source.crates-io]\n"
                    "replace-with = 'vendored-sources'\n"
                    "registry = 'https://example.com'\n"
                    "\n"
                    "[source.vendored-sources]\n"
                    "directory = '{}/vendor'\n"
                    .format(self.rust_root))
        else:
            if os.path.exists('.cargo'):
                shutil.rmtree('.cargo')

    def ensure_vendored(self):
        """Ensure that the vendored sources are available if needed"""
        vendor_dir = os.path.join(self.rust_root, 'vendor')
        # Note that this does not handle updating the vendored dependencies if
        # the rust git repository is updated. Normal development usually does
        # not use vendoring, so hopefully this isn't too much of a problem.
        if self.use_vendored_sources and not os.path.exists(vendor_dir):
            run([
                self.cargo(),
                "vendor",
                "--sync=./src/tools/rust-analyzer/Cargo.toml",
                "--sync=./compiler/rustc_codegen_cranelift/Cargo.toml",
            ], verbose=self.verbose, cwd=self.rust_root)


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
    parser.add_argument('--build')
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('-v', '--verbose', action='count', default=0)

    args = [a for a in sys.argv if a != '-h' and a != '--help']
    args, _ = parser.parse_known_args(args)

    # Configure initial bootstrap
    build = RustBuild()
    build.rust_root = os.path.abspath(os.path.join(__file__, '../../..'))
    build.verbose = args.verbose
    build.clean = args.clean

    # Read from `RUST_BOOTSTRAP_CONFIG`, then `--config`, then fallback to `config.toml` (if it
    # exists).
    toml_path = os.getenv('RUST_BOOTSTRAP_CONFIG') or args.config
    if not toml_path and os.path.exists('config.toml'):
        toml_path = 'config.toml'

    if toml_path:
        if not os.path.exists(toml_path):
            toml_path = os.path.join(build.rust_root, toml_path)

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

    build_dir = build.get_toml('build-dir', 'build') or 'build'
    build.build_dir = os.path.abspath(build_dir.replace("$ROOT", build.rust_root))

    data = stage0_data(build.rust_root)
    build.date = data['date']
    build.rustc_channel = data['rustc']
    build.cargo_channel = data['cargo']

    if "rustfmt" in data:
        build.rustfmt_channel = data['rustfmt']

    if 'dev' in data:
        build.set_dev_environment()
    else:
        build.set_normal_environment()

    build.update_submodules()

    # Fetch/build the bootstrap
    build.build = args.build or build.build_triple()
    build.download_stage0()
    sys.stdout.flush()
    build.ensure_vendored()
    build.build_bootstrap()
    sys.stdout.flush()

    # Run the bootstrap
    args = [build.bootstrap_binary()]
    args.extend(sys.argv[1:])
    env = os.environ.copy()
    env["BOOTSTRAP_PARENT_ID"] = str(os.getpid())
    env["BOOTSTRAP_PYTHON"] = sys.executable
    env["BUILD_DIR"] = build.build_dir
    env["RUSTC_BOOTSTRAP"] = '1'
    if toml_path:
        env["BOOTSTRAP_CONFIG"] = toml_path
    run(args, env=env, verbose=build.verbose)


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
