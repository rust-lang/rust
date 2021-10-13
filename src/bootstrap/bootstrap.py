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

def get(base, url, path, checksums, verbose=False, do_verify=True):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name

    try:
        if do_verify:
            if url not in checksums:
                raise RuntimeError("src/stage0.json doesn't contain a checksum for {}".format(url))
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
        if do_verify and not verify(temp_path, sha256, verbose):
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
    if probably_big or verbose:
        print("downloading {}".format(url))
    # see https://serverfault.com/questions/301128/how-to-download
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
            print("detected default triple {}".format(triple))
        return triple
    except Exception as e:
        if verbose:
            print("rustup not detected: {}".format(e))
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
        self.stage0_rustfmt = None
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
        self.rustc_commit = None

    def download_toolchain(self, stage0=True, rustc_channel=None):
        """Fetch the build system for Rust, written in Rust

        This method will build a cache directory, then it will fetch the
        tarball which has the stage0 compiler used to then bootstrap the Rust
        compiler itself.

        Each downloaded tarball is extracted, after that, the script
        will move all the content to the right place.
        """
        if rustc_channel is None:
            rustc_channel = self.stage0_compiler.version
        bin_root = self.bin_root(stage0)

        key = self.stage0_compiler.date
        if not stage0:
            key += str(self.rustc_commit)
        if self.rustc(stage0).startswith(bin_root) and \
                (not os.path.exists(self.rustc(stage0)) or
                 self.program_out_of_date(self.rustc_stamp(stage0), key)):
            if os.path.exists(bin_root):
                shutil.rmtree(bin_root)
            tarball_suffix = '.tar.xz' if support_xz() else '.tar.gz'
            filename = "rust-std-{}-{}{}".format(
                rustc_channel, self.build, tarball_suffix)
            pattern = "rust-std-{}".format(self.build)
            self._download_component_helper(filename, pattern, tarball_suffix, stage0)
            filename = "rustc-{}-{}{}".format(rustc_channel, self.build,
                                              tarball_suffix)
            self._download_component_helper(filename, "rustc", tarball_suffix, stage0)
            # download-rustc doesn't need its own cargo, it can just use beta's.
            if stage0:
                filename = "cargo-{}-{}{}".format(rustc_channel, self.build,
                                                tarball_suffix)
                self._download_component_helper(filename, "cargo", tarball_suffix)
                self.fix_bin_or_dylib("{}/bin/cargo".format(bin_root))
            else:
                filename = "rustc-dev-{}-{}{}".format(rustc_channel, self.build, tarball_suffix)
                self._download_component_helper(
                    filename, "rustc-dev", tarball_suffix, stage0
                )

            self.fix_bin_or_dylib("{}/bin/rustc".format(bin_root))
            self.fix_bin_or_dylib("{}/bin/rustdoc".format(bin_root))
            lib_dir = "{}/lib".format(bin_root)
            for lib in os.listdir(lib_dir):
                if lib.endswith(".so"):
                    self.fix_bin_or_dylib(os.path.join(lib_dir, lib))
            with output(self.rustc_stamp(stage0)) as rust_stamp:
                rust_stamp.write(key)

        if self.rustfmt() and self.rustfmt().startswith(bin_root) and (
            not os.path.exists(self.rustfmt())
            or self.program_out_of_date(
                self.rustfmt_stamp(),
                "" if self.stage0_rustfmt is None else self.stage0_rustfmt.channel()
            )
        ):
            if self.stage0_rustfmt is not None:
                tarball_suffix = '.tar.xz' if support_xz() else '.tar.gz'
                filename = "rustfmt-{}-{}{}".format(
                    self.stage0_rustfmt.version, self.build, tarball_suffix,
                )
                self._download_component_helper(
                    filename, "rustfmt-preview", tarball_suffix, key=self.stage0_rustfmt.date
                )
                self.fix_bin_or_dylib("{}/bin/rustfmt".format(bin_root))
                self.fix_bin_or_dylib("{}/bin/cargo-fmt".format(bin_root))
                with output(self.rustfmt_stamp()) as rustfmt_stamp:
                    rustfmt_stamp.write(self.stage0_rustfmt.channel())

        # Avoid downloading LLVM twice (once for stage0 and once for the master rustc)
        if self.downloading_llvm() and stage0:
            # We want the most recent LLVM submodule update to avoid downloading
            # LLVM more often than necessary.
            #
            # This git command finds that commit SHA, looking for bors-authored
            # commits that modified src/llvm-project or other relevant version
            # stamp files.
            #
            # This works even in a repository that has not yet initialized
            # submodules.
            top_level = subprocess.check_output([
                "git", "rev-parse", "--show-toplevel",
            ]).decode(sys.getdefaultencoding()).strip()
            llvm_sha = subprocess.check_output([
                "git", "rev-list", "--author=bors@rust-lang.org", "-n1",
                "--first-parent", "HEAD",
                "--",
                "{}/src/llvm-project".format(top_level),
                "{}/src/bootstrap/download-ci-llvm-stamp".format(top_level),
                # the LLVM shared object file is named `LLVM-12-rust-{version}-nightly`
                "{}/src/version".format(top_level)
            ]).decode(sys.getdefaultencoding()).strip()
            llvm_assertions = self.get_toml('assertions', 'llvm') == 'true'
            llvm_root = self.llvm_root()
            llvm_lib = os.path.join(llvm_root, "lib")
            if self.program_out_of_date(self.llvm_stamp(), llvm_sha + str(llvm_assertions)):
                self._download_ci_llvm(llvm_sha, llvm_assertions)
                for binary in ["llvm-config", "FileCheck"]:
                    self.fix_bin_or_dylib(os.path.join(llvm_root, "bin", binary))
                for lib in os.listdir(llvm_lib):
                    if lib.endswith(".so"):
                        self.fix_bin_or_dylib(os.path.join(llvm_lib, lib))
                with output(self.llvm_stamp()) as llvm_stamp:
                    llvm_stamp.write(llvm_sha + str(llvm_assertions))

    def downloading_llvm(self):
        opt = self.get_toml('download-ci-llvm', 'llvm')
        # This is currently all tier 1 targets (since others may not have CI
        # artifacts)
        # https://doc.rust-lang.org/rustc/platform-support.html#tier-1
        supported_platforms = [
            "aarch64-unknown-linux-gnu",
            "i686-pc-windows-gnu",
            "i686-pc-windows-msvc",
            "i686-unknown-linux-gnu",
            "x86_64-unknown-linux-gnu",
            "x86_64-apple-darwin",
            "x86_64-pc-windows-gnu",
            "x86_64-pc-windows-msvc",
        ]
        return opt == "true" \
            or (opt == "if-available" and self.build in supported_platforms)

    def _download_component_helper(
        self, filename, pattern, tarball_suffix, stage0=True, key=None
    ):
        if key is None:
            if stage0:
                key = self.stage0_compiler.date
            else:
                key = self.rustc_commit
        cache_dst = os.path.join(self.build_dir, "cache")
        rustc_cache = os.path.join(cache_dst, key)
        if not os.path.exists(rustc_cache):
            os.makedirs(rustc_cache)

        if stage0:
            base = self._download_url
            url = "dist/{}".format(key)
        else:
            base = "https://ci-artifacts.rust-lang.org"
            url = "rustc-builds/{}".format(self.rustc_commit)
        tarball = os.path.join(rustc_cache, filename)
        if not os.path.exists(tarball):
            get(
                base,
                "{}/{}".format(url, filename),
                tarball,
                self.checksums_sha256,
                verbose=self.verbose,
                do_verify=stage0,
            )
        unpack(tarball, tarball_suffix, self.bin_root(stage0), match=pattern, verbose=self.verbose)

    def _download_ci_llvm(self, llvm_sha, llvm_assertions):
        if not llvm_sha:
            print("error: could not find commit hash for downloading LLVM")
            print("help: maybe your repository history is too shallow?")
            print("help: consider disabling `download-ci-llvm`")
            print("help: or fetch enough history to include one upstream commit")
            exit(1)
        cache_prefix = "llvm-{}-{}".format(llvm_sha, llvm_assertions)
        cache_dst = os.path.join(self.build_dir, "cache")
        rustc_cache = os.path.join(cache_dst, cache_prefix)
        if not os.path.exists(rustc_cache):
            os.makedirs(rustc_cache)

        base = "https://ci-artifacts.rust-lang.org"
        url = "rustc-builds/{}".format(llvm_sha)
        if llvm_assertions:
            url = url.replace('rustc-builds', 'rustc-builds-alt')
        # ci-artifacts are only stored as .xz, not .gz
        if not support_xz():
            print("error: XZ support is required to download LLVM")
            print("help: consider disabling `download-ci-llvm` or using python3")
            exit(1)
        tarball_suffix = '.tar.xz'
        filename = "rust-dev-nightly-" + self.build + tarball_suffix
        tarball = os.path.join(rustc_cache, filename)
        if not os.path.exists(tarball):
            get(
                base,
                "{}/{}".format(url, filename),
                tarball,
                self.checksums_sha256,
                verbose=self.verbose,
                do_verify=False,
            )
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

        # If the user has asked binaries to be patched for Nix, then
        # don't check for NixOS or `/lib`, just continue to the patching.
        if self.get_toml('patch-binaries-for-nix', 'build') != 'true':
            # Use `/etc/os-release` instead of `/etc/NIXOS`.
            # The latter one does not exist on NixOS when using tmpfs as root.
            try:
                with open("/etc/os-release", "r") as f:
                    if not any(line.strip() == "ID=nixos" for line in f):
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

    # If `download-rustc` is set, download the most recent commit with CI artifacts
    def maybe_download_ci_toolchain(self):
        # If `download-rustc` is not set, default to rebuilding.
        download_rustc = self.get_toml("download-rustc", section="rust")
        if download_rustc is None or download_rustc == "false":
            return None
        assert download_rustc == "true" or download_rustc == "if-unchanged", download_rustc

        # Handle running from a directory other than the top level
        rev_parse = ["git", "rev-parse", "--show-toplevel"]
        top_level = subprocess.check_output(rev_parse, universal_newlines=True).strip()
        compiler = "{}/compiler/".format(top_level)
        library = "{}/library/".format(top_level)

        # Look for a version to compare to based on the current commit.
        # Only commits merged by bors will have CI artifacts.
        merge_base = [
            "git", "rev-list", "--author=bors@rust-lang.org", "-n1",
            "--first-parent", "HEAD"
        ]
        commit = subprocess.check_output(merge_base, universal_newlines=True).strip()
        if not commit:
            print("error: could not find commit hash for downloading rustc")
            print("help: maybe your repository history is too shallow?")
            print("help: consider disabling `download-rustc`")
            print("help: or fetch enough history to include one upstream commit")
            exit(1)

        # Warn if there were changes to the compiler or standard library since the ancestor commit.
        status = subprocess.call(["git", "diff-index", "--quiet", commit, "--", compiler, library])
        if status != 0:
            if download_rustc == "if-unchanged":
                return None
            print("warning: `download-rustc` is enabled, but there are changes to \
                   compiler/ or library/")

        if self.verbose:
            print("using downloaded stage1 artifacts from CI (commit {})".format(commit))
        self.rustc_commit = commit
        # FIXME: support downloading artifacts from the beta channel
        self.download_toolchain(False, "nightly")

    def rustc_stamp(self, stage0):
        """Return the path for .rustc-stamp at the given stage

        >>> rb = RustBuild()
        >>> rb.build_dir = "build"
        >>> rb.rustc_stamp(True) == os.path.join("build", "stage0", ".rustc-stamp")
        True
        >>> rb.rustc_stamp(False) == os.path.join("build", "ci-rustc", ".rustc-stamp")
        True
        """
        return os.path.join(self.bin_root(stage0), '.rustc-stamp')

    def rustfmt_stamp(self):
        """Return the path for .rustfmt-stamp

        >>> rb = RustBuild()
        >>> rb.build_dir = "build"
        >>> rb.rustfmt_stamp() == os.path.join("build", "stage0", ".rustfmt-stamp")
        True
        """
        return os.path.join(self.bin_root(True), '.rustfmt-stamp')

    def llvm_stamp(self):
        """Return the path for .rustfmt-stamp

        >>> rb = RustBuild()
        >>> rb.build_dir = "build"
        >>> rb.llvm_stamp() == os.path.join("build", "ci-llvm", ".llvm-stamp")
        True
        """
        return os.path.join(self.llvm_root(), '.llvm-stamp')


    def program_out_of_date(self, stamp_path, key):
        """Check if the given program stamp is out of date"""
        if not os.path.exists(stamp_path) or self.clean:
            return True
        with open(stamp_path, 'r') as stamp:
            return key != stamp.read()

    def bin_root(self, stage0):
        """Return the binary root directory for the given stage

        >>> rb = RustBuild()
        >>> rb.build_dir = "build"
        >>> rb.bin_root(True) == os.path.join("build", "stage0")
        True
        >>> rb.bin_root(False) == os.path.join("build", "ci-rustc")
        True

        When the 'build' property is given should be a nested directory:

        >>> rb.build = "devel"
        >>> rb.bin_root(True) == os.path.join("build", "devel", "stage0")
        True
        """
        if stage0:
            subdir = "stage0"
        else:
            subdir = "ci-rustc"
        return os.path.join(self.build_dir, self.build, subdir)

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

    def rustc(self, stage0):
        """Return config path for rustc"""
        return self.program_config('rustc', stage0)

    def rustfmt(self):
        """Return config path for rustfmt"""
        if self.stage0_rustfmt is None:
            return None
        return self.program_config('rustfmt')

    def program_config(self, program, stage0=True):
        """Return config path for the given program at the given stage

        >>> rb = RustBuild()
        >>> rb.config_toml = 'rustc = "rustc"\\n'
        >>> rb.program_config('rustc')
        'rustc'
        >>> rb.config_toml = ''
        >>> cargo_path = rb.program_config('cargo', True)
        >>> cargo_path.rstrip(".exe") == os.path.join(rb.bin_root(True),
        ... "bin", "cargo")
        True
        >>> cargo_path = rb.program_config('cargo', False)
        >>> cargo_path.rstrip(".exe") == os.path.join(rb.bin_root(False),
        ... "bin", "cargo")
        True
        """
        config = self.get_toml(program)
        if config:
            return os.path.expanduser(config)
        return os.path.join(self.bin_root(stage0), "bin", "{}{}".format(
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
        env["RUSTC"] = self.rustc(True)
        env["LD_LIBRARY_PATH"] = os.path.join(self.bin_root(True), "lib") + \
            (os.pathsep + env["LD_LIBRARY_PATH"]) \
            if "LD_LIBRARY_PATH" in env else ""
        env["DYLD_LIBRARY_PATH"] = os.path.join(self.bin_root(True), "lib") + \
            (os.pathsep + env["DYLD_LIBRARY_PATH"]) \
            if "DYLD_LIBRARY_PATH" in env else ""
        env["LIBRARY_PATH"] = os.path.join(self.bin_root(True), "lib") + \
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

        env["PATH"] = os.path.join(self.bin_root(True), "bin") + \
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
        """Build triple as in LLVM

        Note that `default_build_triple` is moderately expensive,
        so use `self.build` where possible.
        """
        config = self.get_toml('build')
        if config:
            return config
        return default_build_triple(self.verbose)

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

        update_args = ["git", "submodule", "update", "--init", "--recursive", "--depth=1"]
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
            print('Unconditionally updating submodules')
        else:
            print('Updating only changed submodules')
        default_encoding = sys.getdefaultencoding()
        # Only update submodules that are needed to build bootstrap.  These are needed because Cargo
        # currently requires everything in a workspace to be "locally present" when starting a
        # build, and will give a hard error if any Cargo.toml files are missing.
        # FIXME: Is there a way to avoid cloning these eagerly? Bootstrap itself doesn't need to
        #   share a workspace with any tools - maybe it could be excluded from the workspace?
        #   That will still require cloning the submodules the second you check the standard
        #   library, though...
        # FIXME: Is there a way to avoid hard-coding the submodules required?
        # WARNING: keep this in sync with the submodules hard-coded in bootstrap/lib.rs
        submodules = [
            "src/tools/rust-installer",
            "src/tools/cargo",
            "src/tools/rls",
            "src/tools/miri",
            "library/backtrace",
            "library/stdarch"
        ]
        filtered_submodules = []
        submodules_names = []
        for module in submodules:
            check = self.check_submodule(module, slow_submodules)
            filtered_submodules.append((module, check))
            submodules_names.append(module)
        recorded = subprocess.Popen(["git", "ls-tree", "HEAD"] + submodules_names,
                                    cwd=self.rust_root, stdout=subprocess.PIPE)
        recorded = recorded.communicate()[0].decode(default_encoding).strip().splitlines()
        # { filename: hash }
        recorded_submodules = {}
        for data in recorded:
            # [mode, kind, hash, filename]
            data = data.split()
            recorded_submodules[data[3]] = data[2]
        for module in filtered_submodules:
            self.update_submodule(module[0], module[1], recorded_submodules)
        print("Submodules updated in %.2f seconds" % (time() - start_time))

    def set_dist_environment(self, url):
        """Set download URL for normal environment"""
        if 'RUSTUP_DIST_SERVER' in os.environ:
            self._download_url = os.environ['RUSTUP_DIST_SERVER']
        else:
            self._download_url = url

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

    with open(os.path.join(build.rust_root, "src", "stage0.json")) as f:
        data = json.load(f)
    build.checksums_sha256 = data["checksums_sha256"]
    build.stage0_compiler = Stage0Toolchain(data["compiler"])
    if data.get("rustfmt") is not None:
        build.stage0_rustfmt = Stage0Toolchain(data["rustfmt"])

    build.set_dist_environment(data["dist_server"])

    build.build = args.build or build.build_triple()
    build.update_submodules()

    # Fetch/build the bootstrap
    build.download_toolchain()
    # Download the master compiler if `download-rustc` is set
    build.maybe_download_ci_toolchain()
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
    if build.rustc_commit is not None:
        env["BOOTSTRAP_DOWNLOAD_RUSTC"] = '1'
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
