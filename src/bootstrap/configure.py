#!/usr/bin/env python

# ignore-tidy-linelength

from __future__ import absolute_import, division, print_function
import shlex
import sys
import os

rust_dir = os.path.dirname(os.path.abspath(__file__))
rust_dir = os.path.dirname(rust_dir)
rust_dir = os.path.dirname(rust_dir)
sys.path.append(os.path.join(rust_dir, "src", "bootstrap"))
import bootstrap  # noqa: E402


class Option(object):
    def __init__(self, name, rustbuild, desc, value):
        self.name = name
        self.rustbuild = rustbuild
        self.desc = desc
        self.value = value


options = []


def o(*args):
    options.append(Option(*args, value=False))


def v(*args):
    options.append(Option(*args, value=True))


o(
    "debug",
    "rust.debug",
    "enables debugging environment; does not affect optimization of bootstrapped code",
)
o("docs", "build.docs", "build standard library documentation")
o("compiler-docs", "build.compiler-docs", "build compiler documentation")
o("optimize-tests", "rust.optimize-tests", "build tests with optimizations")
o("verbose-tests", "rust.verbose-tests", "enable verbose output when running tests")
o(
    "ccache",
    "build.ccache",
    "invoke gcc/clang/rustc via ccache to reuse object files between builds",
)
o(
    "sccache",
    None,
    "invoke gcc/clang/rustc via sccache to reuse object files between builds",
)
o("local-rust", None, "use an installed rustc rather than downloading a snapshot")
v("local-rust-root", None, "set prefix for local rust binary")
o(
    "local-rebuild",
    "build.local-rebuild",
    "assume local-rust matches the current version, for rebuilds; implies local-rust, and is implied if local-rust already matches the current version",
)
o(
    "llvm-static-stdcpp",
    "llvm.static-libstdcpp",
    "statically link to libstdc++ for LLVM",
)
o(
    "llvm-link-shared",
    "llvm.link-shared",
    "prefer shared linking to LLVM (llvm-config --link-shared)",
)
o("rpath", "rust.rpath", "build rpaths into rustc itself")
o("codegen-tests", "rust.codegen-tests", "run the tests/codegen tests")
o(
    "ninja",
    "llvm.ninja",
    "build LLVM using the Ninja generator (for MSVC, requires building in the correct environment)",
)
o("locked-deps", "build.locked-deps", "force Cargo.lock to be up to date")
o("vendor", "build.vendor", "enable usage of vendored Rust crates")
o(
    "sanitizers",
    "build.sanitizers",
    "build the sanitizer runtimes (asan, dfsan, lsan, msan, tsan, hwasan)",
)
o(
    "dist-src",
    "rust.dist-src",
    "when building tarballs enables building a source tarball",
)
o(
    "cargo-native-static",
    "build.cargo-native-static",
    "static native libraries in cargo",
)
o("profiler", "build.profiler", "build the profiler runtime")
o("full-tools", None, "enable all tools")
o("lld", "rust.lld", "build lld")
o("llvm-bitcode-linker", "rust.llvm-bitcode-linker", "build llvm bitcode linker")
o("clang", "llvm.clang", "build clang")
o("use-libcxx", "llvm.use-libcxx", "build LLVM with libc++")
o("control-flow-guard", "rust.control-flow-guard", "Enable Control Flow Guard")
o(
    "patch-binaries-for-nix",
    "build.patch-binaries-for-nix",
    "whether patch binaries for usage with Nix toolchains",
)
o("new-symbol-mangling", "rust.new-symbol-mangling", "use symbol-mangling-version v0")

v("llvm-cflags", "llvm.cflags", "build LLVM with these extra compiler flags")
v("llvm-cxxflags", "llvm.cxxflags", "build LLVM with these extra compiler flags")
v("llvm-ldflags", "llvm.ldflags", "build LLVM with these extra linker flags")

v("llvm-libunwind", "rust.llvm-libunwind", "use LLVM libunwind")

# Optimization and debugging options. These may be overridden by the release
# channel, etc.
o("optimize-llvm", "llvm.optimize", "build optimized LLVM")
o("llvm-assertions", "llvm.assertions", "build LLVM with assertions")
o("llvm-enzyme", "llvm.enzyme", "build LLVM with enzyme")
o("llvm-offload", "llvm.offload", "build LLVM with gpu offload support")
o("llvm-plugins", "llvm.plugins", "build LLVM with plugin interface")
o("debug-assertions", "rust.debug-assertions", "build with debugging assertions")
o(
    "debug-assertions-std",
    "rust.debug-assertions-std",
    "build the standard library with debugging assertions",
)
o("overflow-checks", "rust.overflow-checks", "build with overflow checks")
o(
    "overflow-checks-std",
    "rust.overflow-checks-std",
    "build the standard library with overflow checks",
)
o(
    "llvm-release-debuginfo",
    "llvm.release-debuginfo",
    "build LLVM with debugger metadata",
)
v("debuginfo-level", "rust.debuginfo-level", "debuginfo level for Rust code")
v(
    "debuginfo-level-rustc",
    "rust.debuginfo-level-rustc",
    "debuginfo level for the compiler",
)
v(
    "debuginfo-level-std",
    "rust.debuginfo-level-std",
    "debuginfo level for the standard library",
)
v(
    "debuginfo-level-tools",
    "rust.debuginfo-level-tools",
    "debuginfo level for the tools",
)
v(
    "debuginfo-level-tests",
    "rust.debuginfo-level-tests",
    "debuginfo level for the test suites run with compiletest",
)
v(
    "save-toolstates",
    "rust.save-toolstates",
    "save build and test status of external tools into this file",
)

v("prefix", "install.prefix", "set installation prefix")
v("localstatedir", "install.localstatedir", "local state directory")
v("datadir", "install.datadir", "install data")
v("sysconfdir", "install.sysconfdir", "install system configuration files")
v("infodir", "install.infodir", "install additional info")
v("libdir", "install.libdir", "install libraries")
v("mandir", "install.mandir", "install man pages in PATH")
v("docdir", "install.docdir", "install documentation in PATH")
v("bindir", "install.bindir", "install binaries")

v("llvm-root", None, "set LLVM root")
v("llvm-config", None, "set path to llvm-config")
v("llvm-filecheck", None, "set path to LLVM's FileCheck utility")
v("python", "build.python", "set path to python")
v("android-ndk", "build.android-ndk", "set path to Android NDK")
v(
    "musl-root",
    "target.x86_64-unknown-linux-musl.musl-root",
    "MUSL root installation directory (deprecated)",
)
v(
    "musl-root-x86_64",
    "target.x86_64-unknown-linux-musl.musl-root",
    "x86_64-unknown-linux-musl install directory",
)
v(
    "musl-root-i586",
    "target.i586-unknown-linux-musl.musl-root",
    "i586-unknown-linux-musl install directory",
)
v(
    "musl-root-i686",
    "target.i686-unknown-linux-musl.musl-root",
    "i686-unknown-linux-musl install directory",
)
v(
    "musl-root-arm",
    "target.arm-unknown-linux-musleabi.musl-root",
    "arm-unknown-linux-musleabi install directory",
)
v(
    "musl-root-armhf",
    "target.arm-unknown-linux-musleabihf.musl-root",
    "arm-unknown-linux-musleabihf install directory",
)
v(
    "musl-root-armv5te",
    "target.armv5te-unknown-linux-musleabi.musl-root",
    "armv5te-unknown-linux-musleabi install directory",
)
v(
    "musl-root-armv7",
    "target.armv7-unknown-linux-musleabi.musl-root",
    "armv7-unknown-linux-musleabi install directory",
)
v(
    "musl-root-armv7hf",
    "target.armv7-unknown-linux-musleabihf.musl-root",
    "armv7-unknown-linux-musleabihf install directory",
)
v(
    "musl-root-aarch64",
    "target.aarch64-unknown-linux-musl.musl-root",
    "aarch64-unknown-linux-musl install directory",
)
v(
    "musl-root-mips",
    "target.mips-unknown-linux-musl.musl-root",
    "mips-unknown-linux-musl install directory",
)
v(
    "musl-root-mipsel",
    "target.mipsel-unknown-linux-musl.musl-root",
    "mipsel-unknown-linux-musl install directory",
)
v(
    "musl-root-mips64",
    "target.mips64-unknown-linux-muslabi64.musl-root",
    "mips64-unknown-linux-muslabi64 install directory",
)
v(
    "musl-root-mips64el",
    "target.mips64el-unknown-linux-muslabi64.musl-root",
    "mips64el-unknown-linux-muslabi64 install directory",
)
v(
    "musl-root-powerpc64le",
    "target.powerpc64le-unknown-linux-musl.musl-root",
    "powerpc64le-unknown-linux-musl install directory",
)
v(
    "musl-root-riscv32gc",
    "target.riscv32gc-unknown-linux-musl.musl-root",
    "riscv32gc-unknown-linux-musl install directory",
)
v(
    "musl-root-riscv64gc",
    "target.riscv64gc-unknown-linux-musl.musl-root",
    "riscv64gc-unknown-linux-musl install directory",
)
v(
    "musl-root-loongarch64",
    "target.loongarch64-unknown-linux-musl.musl-root",
    "loongarch64-unknown-linux-musl install directory",
)
v(
    "qemu-armhf-rootfs",
    "target.arm-unknown-linux-gnueabihf.qemu-rootfs",
    "rootfs in qemu testing, you probably don't want to use this",
)
v(
    "qemu-aarch64-rootfs",
    "target.aarch64-unknown-linux-gnu.qemu-rootfs",
    "rootfs in qemu testing, you probably don't want to use this",
)
v(
    "qemu-riscv64-rootfs",
    "target.riscv64gc-unknown-linux-gnu.qemu-rootfs",
    "rootfs in qemu testing, you probably don't want to use this",
)
v(
    "experimental-targets",
    "llvm.experimental-targets",
    "experimental LLVM targets to build",
)
v("release-channel", "rust.channel", "the name of the release channel to build")
v(
    "release-description",
    "rust.description",
    "optional descriptive string for version output",
)
v("dist-compression-formats", None, "List of compression formats to use")

# Used on systems where "cc" is unavailable
v("default-linker", "rust.default-linker", "the default linker")

# Many of these are saved below during the "writing configuration" step
# (others are conditionally saved).
o("manage-submodules", "build.submodules", "let the build manage the git submodules")
o(
    "full-bootstrap",
    "build.full-bootstrap",
    "build three compilers instead of two (not recommended except for testing reproducible builds)",
)
o("extended", "build.extended", "build an extended rust tool set")

v("bootstrap-cache-path", None, "use provided path for the bootstrap cache")
v("tools", None, "List of extended tools will be installed")
v("codegen-backends", None, "List of codegen backends to build")
v("build", "build.build", "GNUs ./configure syntax LLVM build triple")
v("host", None, "List of GNUs ./configure syntax LLVM host triples")
v("target", None, "List of GNUs ./configure syntax LLVM target triples")

# Options specific to this configure script
o(
    "option-checking",
    None,
    "complain about unrecognized options in this configure script",
)
o(
    "verbose-configure",
    None,
    "don't truncate options when printing them in this configure script",
)
v("set", None, "set arbitrary key/value pairs in TOML configuration")


def p(msg):
    print("configure: " + msg)


def err(msg):
    print("\nconfigure: ERROR: " + msg + "\n")
    sys.exit(1)


def is_value_list(key):
    for option in options:
        if option.name == key and option.desc.startswith("List of"):
            return True
    return False


if "--help" in sys.argv or "-h" in sys.argv:
    print("Usage: ./configure [options]")
    print("")
    print("Options")
    for option in options:
        if "android" in option.name:
            # no one needs to know about these obscure options
            continue
        if option.value:
            print("\t{:30} {}".format("--{}=VAL".format(option.name), option.desc))
        else:
            print("\t--enable-{:25} OR --disable-{}".format(option.name, option.name))
            print("\t\t" + option.desc)
    print("")
    print("This configure script is a thin configuration shim over the true")
    print("configuration system, `config.toml`. You can explore the comments")
    print("in `config.example.toml` next to this configure script to see")
    print("more information about what each option is. Additionally you can")
    print("pass `--set` as an argument to set arbitrary key/value pairs")
    print("in the TOML configuration if desired")
    print("")
    print("Also note that all options which take `--enable` can similarly")
    print("be passed with `--disable-foo` to forcibly disable the option")
    sys.exit(0)

VERBOSE = False


# Parse command line arguments into a valid build configuration.
def parse_args(args):
    known_args = validate_args(args)
    config = generate_config(known_args)
    if "profile" not in config:
        _set("profile", "dist", config)
    _set("build.configure-args", args, config)
    return config


# Validate command line arguments, throwing an error if there are any unknown
# arguments, missing values or duplicate arguments when option-checking is also
# passed as an argument. Returns a dictionary of known arguments.
def validate_args(args):
    unknown_args = []
    need_value_args = []
    duplicate_args = []
    known_args = {}

    i = 0
    while i < len(args):
        arg = args[i]
        i += 1
        if not arg.startswith("--"):
            unknown_args.append(arg)
            continue

        found = False
        for option in options:
            value = None
            if option.value:
                keyval = arg[2:].split("=", 1)
                key = keyval[0]
                if option.name != key:
                    continue
                if option.name == "infodir" or option.name == "localstatedir":
                    # These are used by rpm, but aren't accepted by x.py.
                    # Give a warning that they're ignored, but not a hard error.
                    p("WARNING: {} will be ignored".format(option.name))

                if len(keyval) > 1:
                    value = keyval[1]
                elif i < len(args):
                    value = args[i]
                    i += 1
                else:
                    need_value_args.append(arg)
                    continue
            else:
                if arg[2:] == "enable-" + option.name:
                    value = True
                elif arg[2:] == "disable-" + option.name:
                    value = False
                else:
                    continue
            found = True

            if option.name not in known_args:
                known_args[option.name] = []
            elif option.name in known_args and option.name != "set":
                duplicate_args.append(option.name)

            known_args[option.name].append((option, value))
            break

        if not found:
            unknown_args.append(arg)

    # NOTE: here and a few other places, we use [-1] to apply the *last* value
    # passed.  But if option-checking is enabled, then the known_args loop will
    # also assert that options are only passed once.
    option_checking = (
        "option-checking" not in known_args or known_args["option-checking"][-1][1]
    )
    if option_checking:
        if len(unknown_args) > 0:
            err("Option '" + unknown_args[0] + "' is not recognized")
        if len(need_value_args) > 0:
            err("Option '{0}' needs a value ({0}=val)".format(need_value_args[0]))
        for key, values in known_args.items():
            if len(values) > 1 and key != "set":
                err("Option '{}' provided more than once".format(key))

    global VERBOSE
    VERBOSE = "verbose-configure" in known_args

    return known_args


def _set(key, value, config):
    if isinstance(value, list):
        # Remove empty values, which value.split(',') tends to generate and
        # replace single quotes for double quotes to ensure correct parsing.
        value = [v.replace("'", '"') for v in value if v]

    s = "{:20} := {}".format(key, value)
    if len(s) < 70 or VERBOSE:
        p(s)
    else:
        p(s[:70] + " ...")

    arr = config

    # Split `key` on periods using shell semantics.
    lexer = shlex.shlex(key, posix=True)
    lexer.whitespace = "."
    lexer.wordchars += "-"
    parts = list(lexer)

    for i, part in enumerate(parts):
        if i == len(parts) - 1:
            if is_value_list(part) and isinstance(value, str):
                value = value.split(",")
            arr[part] = value
        else:
            if part not in arr:
                arr[part] = {}
            arr = arr[part]


# Convert a validated list of command line arguments into configuration
def generate_config(known_args):
    config = {}
    for key in known_args:
        # The `set` option is special and can be passed a bunch of times
        if key == "set":
            for _option, value in known_args[key]:
                keyval = value.split("=", 1)
                if len(keyval) == 1 or keyval[1] == "true":
                    value = True
                elif keyval[1] == "false":
                    value = False
                else:
                    value = keyval[1]
                _set(keyval[0], value, config)
            continue

        arr = known_args[key]
        option, value = arr[-1]

        # If we have a clear avenue to set our value in rustbuild, do so
        if option.rustbuild is not None:
            _set(option.rustbuild, value, config)
            continue

        # Otherwise we're a "special" option and need some extra handling, so do
        # that here.
        if "build" in known_args:
            build_triple = known_args["build"][-1][1]
        else:
            build_triple = bootstrap.default_build_triple(verbose=False)

        if option.name == "sccache":
            _set("build.ccache", "sccache", config)
        elif option.name == "local-rust":
            for path in os.environ["PATH"].split(os.pathsep):
                if os.path.exists(path + "/rustc"):
                    _set("build.rustc", path + "/rustc", config)
                    break
            for path in os.environ["PATH"].split(os.pathsep):
                if os.path.exists(path + "/cargo"):
                    _set("build.cargo", path + "/cargo", config)
                    break
        elif option.name == "local-rust-root":
            _set("build.rustc", value + "/bin/rustc", config)
            _set("build.cargo", value + "/bin/cargo", config)
        elif option.name == "llvm-root":
            _set(
                "target.{}.llvm-config".format(build_triple),
                value + "/bin/llvm-config",
                config,
            )
        elif option.name == "llvm-config":
            _set("target.{}.llvm-config".format(build_triple), value, config)
        elif option.name == "llvm-filecheck":
            _set("target.{}.llvm-filecheck".format(build_triple), value, config)
        elif option.name == "tools":
            _set("build.tools", value.split(","), config)
        elif option.name == "bootstrap-cache-path":
            _set("build.bootstrap-cache-path", value, config)
        elif option.name == "codegen-backends":
            _set("rust.codegen-backends", value.split(","), config)
        elif option.name == "host":
            _set("build.host", value.split(","), config)
        elif option.name == "target":
            _set("build.target", value.split(","), config)
        elif option.name == "full-tools":
            _set("rust.codegen-backends", ["llvm"], config)
            _set("rust.lld", True, config)
            _set("rust.llvm-tools", True, config)
            _set("rust.llvm-bitcode-linker", True, config)
            _set("build.extended", True, config)
        elif option.name in ["option-checking", "verbose-configure"]:
            # this was handled above
            pass
        elif option.name == "dist-compression-formats":
            _set("dist.compression-formats", value.split(","), config)
        else:
            raise RuntimeError("unhandled option {}".format(option.name))
    return config


def get_configured_targets(config):
    targets = set()
    if "build" in config and "build" in config["build"]:
        targets.add(config["build"]["build"])
    else:
        targets.add(bootstrap.default_build_triple(verbose=False))

    if "build" in config:
        if "host" in config["build"]:
            targets.update(config["build"]["host"])
        if "target" in config["build"]:
            targets.update(config["build"]["target"])
    if "target" in config:
        for target in config["target"]:
            targets.add(target)
    return list(targets)


def write_block(f, config, block):
    last_line = block[-1]
    key = last_line.split("=")[0].strip(" #")
    value = config[key] if key in config else None
    if value is not None:
        for ln in block[:-1]:
            f.write(ln + "\n")
        f.write("{} = {}\n".format(key, to_toml(value)))


def write_section(f, config, section_lines):
    block = []
    for line in section_lines[1:]:
        if line.count("=") == 1:
            block.append(line)
            write_block(f, config, block)
            block = []
        else:
            block.append(line)


# Write out the configuration toml file to f, using config.example.toml as a
# template.
def write_config_toml(f, config):
    with open(rust_dir + "/config.example.toml") as example_config:
        lines = example_config.read().split("\n")

    section_name = None
    section = []
    block = []

    i = 0
    # Drop the initial comment block
    for line in lines:
        if not line.startswith("#"):
            break
        i += 1

    for line in lines[i:]:
        if line.startswith("["):
            if section:
                # Write out the previous section before starting a new one.
                #
                # Note that the `target` section is handled separately as we'll
                # duplicate it per configured target, so there's a bit of special
                # handling for that here.
                if section_name.startswith("target"):
                    for target in get_configured_targets(config):
                        # For `.` to be valid TOML, it needs to be quoted. But `bootstrap.py` doesn't
                        # use a proper TOML parser and fails to parse the target.
                        # Avoid using quotes unless it's necessary.
                        target_trip = "'{}'".format(target) if "." in target else target
                        f.write("\n[target.{}]\n".format(target_trip))
                        if "target" in config and target in config["target"]:
                            write_section(f, config["target"][target], section)
                elif "." in section_name:
                    raise RuntimeError(
                        "don't know how to deal with section: {}".format(section_name)
                    )
                else:
                    f.write("\n[{}]\n".format(section_name))
                    if section_name in config:
                        write_section(f, config[section_name], section)
                # Start a new section
                section = []
                section_name = None
            section_name = line[1:-1]
            section.append(line)
        elif section_name is not None:
            section.append(line)
        else:
            # this a top-level configuration
            if line.count("=") == 1:
                block.append(line)
                write_block(f, config, block)
                block = []
            else:
                block.append(line)


def is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


# Here we walk through the constructed configuration we have from the parsed
# command line arguments. We then apply each piece of configuration by
# basically just doing a `sed` to change the various configuration line to what
# we've got configure.
def to_toml(value):
    if isinstance(value, bool):
        if value:
            return "true"
        else:
            return "false"
    elif isinstance(value, list):
        return "[" + ", ".join(map(to_toml, value)) + "]"
    elif isinstance(value, str):
        # Don't put quotes around numeric values
        if is_number(value):
            return value
        else:
            return "'" + value + "'"
    elif isinstance(value, dict):
        return (
            "{"
            + ", ".join(
                map(
                    lambda a: "{} = {}".format(to_toml(a[0]), to_toml(a[1])),
                    value.items(),
                )
            )
            + "}"
        )
    else:
        raise RuntimeError("no toml")


def quit_if_file_exists(file):
    if os.path.isfile(file):
        msg = "Existing '{}' detected. Exiting".format(file)

        # If the output object directory isn't empty, we can get these errors
        host_objdir = os.environ.get("OBJDIR_ON_HOST")
        if host_objdir is not None:
            msg += "\nIs objdir '{}' clean?".format(host_objdir)

        err(msg)


if __name__ == "__main__":
    # If 'config.toml' already exists, exit the script at this point
    quit_if_file_exists("config.toml")

    if "GITHUB_ACTIONS" in os.environ:
        print("::group::Configure the build")
    p("processing command line")
    # Parse all known arguments into a configuration structure that reflects the
    # TOML we're going to write out
    p("")
    config = parse_args(sys.argv[1:])

    # Now that we've built up our `config.toml`, write it all out in the same
    # order that we read it in.
    p("")
    p("writing `config.toml` in current directory")
    with bootstrap.output("config.toml") as f:
        write_config_toml(f, config)

    with bootstrap.output("Makefile") as f:
        contents = os.path.join(rust_dir, "src", "bootstrap", "mk", "Makefile.in")
        contents = open(contents).read()
        contents = contents.replace("$(CFG_SRC_DIR)", rust_dir + "/")
        contents = contents.replace("$(CFG_PYTHON)", sys.executable)
        f.write(contents)

    p("")
    p("run `python {}/x.py --help`".format(rust_dir))
    if "GITHUB_ACTIONS" in os.environ:
        print("::endgroup::")
