#!/usr/bin/env python

# ignore-tidy-linelength

from __future__ import absolute_import, division, print_function
import sys
import os
rust_dir = os.path.dirname(os.path.abspath(__file__))
rust_dir = os.path.dirname(rust_dir)
rust_dir = os.path.dirname(rust_dir)
sys.path.append(os.path.join(rust_dir, "src", "bootstrap"))
import bootstrap


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


o("debug", "rust.debug", "enables debugging environment; does not affect optimization of bootstrapped code")
o("docs", "build.docs", "build standard library documentation")
o("compiler-docs", "build.compiler-docs", "build compiler documentation")
o("optimize-tests", "rust.optimize-tests", "build tests with optimizations")
o("verbose-tests", "rust.verbose-tests", "enable verbose output when running tests")
o("ccache", "llvm.ccache", "invoke gcc/clang via ccache to reuse object files between builds")
o("sccache", None, "invoke gcc/clang via sccache to reuse object files between builds")
o("local-rust", None, "use an installed rustc rather than downloading a snapshot")
v("local-rust-root", None, "set prefix for local rust binary")
o("local-rebuild", "build.local-rebuild", "assume local-rust matches the current version, for rebuilds; implies local-rust, and is implied if local-rust already matches the current version")
o("llvm-static-stdcpp", "llvm.static-libstdcpp", "statically link to libstdc++ for LLVM")
o("llvm-link-shared", "llvm.link-shared", "prefer shared linking to LLVM (llvm-config --link-shared)")
o("rpath", "rust.rpath", "build rpaths into rustc itself")
o("codegen-tests", "rust.codegen-tests", "run the tests/codegen tests")
o("option-checking", None, "complain about unrecognized options in this configure script")
o("ninja", "llvm.ninja", "build LLVM using the Ninja generator (for MSVC, requires building in the correct environment)")
o("locked-deps", "build.locked-deps", "force Cargo.lock to be up to date")
o("vendor", "build.vendor", "enable usage of vendored Rust crates")
o("sanitizers", "build.sanitizers", "build the sanitizer runtimes (asan, lsan, msan, tsan, hwasan)")
o("dist-src", "rust.dist-src", "when building tarballs enables building a source tarball")
o("cargo-native-static", "build.cargo-native-static", "static native libraries in cargo")
o("profiler", "build.profiler", "build the profiler runtime")
o("full-tools", None, "enable all tools")
o("lld", "rust.lld", "build lld")
o("clang", "llvm.clang", "build clang")
o("missing-tools", "dist.missing-tools", "allow failures when building tools")
o("use-libcxx", "llvm.use-libcxx", "build LLVM with libc++")
o("control-flow-guard", "rust.control-flow-guard", "Enable Control Flow Guard")

v("llvm-cflags", "llvm.cflags", "build LLVM with these extra compiler flags")
v("llvm-cxxflags", "llvm.cxxflags", "build LLVM with these extra compiler flags")
v("llvm-ldflags", "llvm.ldflags", "build LLVM with these extra linker flags")

v("llvm-libunwind", "rust.llvm-libunwind", "use LLVM libunwind")

# Optimization and debugging options. These may be overridden by the release
# channel, etc.
o("optimize-llvm", "llvm.optimize", "build optimized LLVM")
o("llvm-assertions", "llvm.assertions", "build LLVM with assertions")
o("llvm-plugins", "llvm.plugins", "build LLVM with plugin interface")
o("debug-assertions", "rust.debug-assertions", "build with debugging assertions")
o("debug-assertions-std", "rust.debug-assertions-std", "build the standard library with debugging assertions")
o("overflow-checks", "rust.overflow-checks", "build with overflow checks")
o("overflow-checks-std", "rust.overflow-checks-std", "build the standard library with overflow checks")
o("llvm-release-debuginfo", "llvm.release-debuginfo", "build LLVM with debugger metadata")
v("debuginfo-level", "rust.debuginfo-level", "debuginfo level for Rust code")
v("debuginfo-level-rustc", "rust.debuginfo-level-rustc", "debuginfo level for the compiler")
v("debuginfo-level-std", "rust.debuginfo-level-std", "debuginfo level for the standard library")
v("debuginfo-level-tools", "rust.debuginfo-level-tools", "debuginfo level for the tools")
v("debuginfo-level-tests", "rust.debuginfo-level-tests", "debuginfo level for the test suites run with compiletest")
v("save-toolstates", "rust.save-toolstates", "save build and test status of external tools into this file")

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
v("android-cross-path", "target.arm-linux-androideabi.android-ndk",
  "Android NDK standalone path (deprecated)")
v("i686-linux-android-ndk", "target.i686-linux-android.android-ndk",
  "i686-linux-android NDK standalone path")
v("arm-linux-androideabi-ndk", "target.arm-linux-androideabi.android-ndk",
  "arm-linux-androideabi NDK standalone path")
v("armv7-linux-androideabi-ndk", "target.armv7-linux-androideabi.android-ndk",
  "armv7-linux-androideabi NDK standalone path")
v("thumbv7neon-linux-androideabi-ndk", "target.thumbv7neon-linux-androideabi.android-ndk",
  "thumbv7neon-linux-androideabi NDK standalone path")
v("aarch64-linux-android-ndk", "target.aarch64-linux-android.android-ndk",
  "aarch64-linux-android NDK standalone path")
v("x86_64-linux-android-ndk", "target.x86_64-linux-android.android-ndk",
  "x86_64-linux-android NDK standalone path")
v("musl-root", "target.x86_64-unknown-linux-musl.musl-root",
  "MUSL root installation directory (deprecated)")
v("musl-root-x86_64", "target.x86_64-unknown-linux-musl.musl-root",
  "x86_64-unknown-linux-musl install directory")
v("musl-root-i586", "target.i586-unknown-linux-musl.musl-root",
  "i586-unknown-linux-musl install directory")
v("musl-root-i686", "target.i686-unknown-linux-musl.musl-root",
  "i686-unknown-linux-musl install directory")
v("musl-root-arm", "target.arm-unknown-linux-musleabi.musl-root",
  "arm-unknown-linux-musleabi install directory")
v("musl-root-armhf", "target.arm-unknown-linux-musleabihf.musl-root",
  "arm-unknown-linux-musleabihf install directory")
v("musl-root-armv5te", "target.armv5te-unknown-linux-musleabi.musl-root",
  "armv5te-unknown-linux-musleabi install directory")
v("musl-root-armv7", "target.armv7-unknown-linux-musleabi.musl-root",
  "armv7-unknown-linux-musleabi install directory")
v("musl-root-armv7hf", "target.armv7-unknown-linux-musleabihf.musl-root",
  "armv7-unknown-linux-musleabihf install directory")
v("musl-root-aarch64", "target.aarch64-unknown-linux-musl.musl-root",
  "aarch64-unknown-linux-musl install directory")
v("musl-root-mips", "target.mips-unknown-linux-musl.musl-root",
  "mips-unknown-linux-musl install directory")
v("musl-root-mipsel", "target.mipsel-unknown-linux-musl.musl-root",
  "mipsel-unknown-linux-musl install directory")
v("musl-root-mips64", "target.mips64-unknown-linux-muslabi64.musl-root",
  "mips64-unknown-linux-muslabi64 install directory")
v("musl-root-mips64el", "target.mips64el-unknown-linux-muslabi64.musl-root",
  "mips64el-unknown-linux-muslabi64 install directory")
v("qemu-armhf-rootfs", "target.arm-unknown-linux-gnueabihf.qemu-rootfs",
  "rootfs in qemu testing, you probably don't want to use this")
v("qemu-aarch64-rootfs", "target.aarch64-unknown-linux-gnu.qemu-rootfs",
  "rootfs in qemu testing, you probably don't want to use this")
v("qemu-riscv64-rootfs", "target.riscv64gc-unknown-linux-gnu.qemu-rootfs",
  "rootfs in qemu testing, you probably don't want to use this")
v("experimental-targets", "llvm.experimental-targets",
  "experimental LLVM targets to build")
v("release-channel", "rust.channel", "the name of the release channel to build")
v("release-description", "rust.description", "optional descriptive string for version output")
v("dist-compression-formats", None,
  "comma-separated list of compression formats to use")

# Used on systems where "cc" is unavailable
v("default-linker", "rust.default-linker", "the default linker")

# Many of these are saved below during the "writing configuration" step
# (others are conditionally saved).
o("manage-submodules", "build.submodules", "let the build manage the git submodules")
o("full-bootstrap", "build.full-bootstrap", "build three compilers instead of two (not recommended except for testing reproducible builds)")
o("extended", "build.extended", "build an extended rust tool set")

v("tools", None, "List of extended tools will be installed")
v("codegen-backends", None, "List of codegen backends to build")
v("build", "build.build", "GNUs ./configure syntax LLVM build triple")
v("host", None, "GNUs ./configure syntax LLVM host triples")
v("target", None, "GNUs ./configure syntax LLVM target triples")

v("set", None, "set arbitrary key/value pairs in TOML configuration")


def p(msg):
    print("configure: " + msg)


def err(msg):
    print("configure: error: " + msg)
    sys.exit(1)


if '--help' in sys.argv or '-h' in sys.argv:
    print('Usage: ./configure [options]')
    print('')
    print('Options')
    for option in options:
        if 'android' in option.name:
            # no one needs to know about these obscure options
            continue
        if option.value:
            print('\t{:30} {}'.format('--{}=VAL'.format(option.name), option.desc))
        else:
            print('\t{:30} {}'.format('--enable-{}'.format(option.name), option.desc))
    print('')
    print('This configure script is a thin configuration shim over the true')
    print('configuration system, `config.toml`. You can explore the comments')
    print('in `config.example.toml` next to this configure script to see')
    print('more information about what each option is. Additionally you can')
    print('pass `--set` as an argument to set arbitrary key/value pairs')
    print('in the TOML configuration if desired')
    print('')
    print('Also note that all options which take `--enable` can similarly')
    print('be passed with `--disable-foo` to forcibly disable the option')
    sys.exit(0)

# Parse all command line arguments into one of these three lists, handling
# boolean and value-based options separately
unknown_args = []
need_value_args = []
known_args = {}

p("processing command line")
i = 1
while i < len(sys.argv):
    arg = sys.argv[i]
    i += 1
    if not arg.startswith('--'):
        unknown_args.append(arg)
        continue

    found = False
    for option in options:
        value = None
        if option.value:
            keyval = arg[2:].split('=', 1)
            key = keyval[0]
            if option.name != key:
                continue

            if len(keyval) > 1:
                value = keyval[1]
            elif i < len(sys.argv):
                value = sys.argv[i]
                i += 1
            else:
                need_value_args.append(arg)
                continue
        else:
            if arg[2:] == 'enable-' + option.name:
                value = True
            elif arg[2:] == 'disable-' + option.name:
                value = False
            else:
                continue

        found = True
        if option.name not in known_args:
            known_args[option.name] = []
        known_args[option.name].append((option, value))
        break

    if not found:
        unknown_args.append(arg)
p("")

# Note: here and a few other places, we use [-1] to apply the *last* value
# passed.  But if option-checking is enabled, then the known_args loop will
# also assert that options are only passed once.
option_checking = ('option-checking' not in known_args
                   or known_args['option-checking'][-1][1])
if option_checking:
    if len(unknown_args) > 0:
        err("Option '" + unknown_args[0] + "' is not recognized")
    if len(need_value_args) > 0:
        err("Option '{0}' needs a value ({0}=val)".format(need_value_args[0]))

# Parse all known arguments into a configuration structure that reflects the
# TOML we're going to write out
config = {}


def build():
    if 'build' in known_args:
        return known_args['build'][-1][1]
    return bootstrap.default_build_triple(verbose=False)


def set(key, value):
    if isinstance(value, list):
        # Remove empty values, which value.split(',') tends to generate.
        value = [v for v in value if v]

    s = "{:20} := {}".format(key, value)
    if len(s) < 70:
        p(s)
    else:
        p(s[:70] + " ...")

    arr = config
    parts = key.split('.')
    for i, part in enumerate(parts):
        if i == len(parts) - 1:
            arr[part] = value
        else:
            if part not in arr:
                arr[part] = {}
            arr = arr[part]


for key in known_args:
    # The `set` option is special and can be passed a bunch of times
    if key == 'set':
        for option, value in known_args[key]:
            keyval = value.split('=', 1)
            if len(keyval) == 1 or keyval[1] == "true":
                value = True
            elif keyval[1] == "false":
                value = False
            else:
                value = keyval[1]
            set(keyval[0], value)
        continue

    # Ensure each option is only passed once
    arr = known_args[key]
    if option_checking and len(arr) > 1:
        err("Option '{}' provided more than once".format(key))
    option, value = arr[-1]

    # If we have a clear avenue to set our value in rustbuild, do so
    if option.rustbuild is not None:
        set(option.rustbuild, value)
        continue

    # Otherwise we're a "special" option and need some extra handling, so do
    # that here.
    if option.name == 'sccache':
        set('llvm.ccache', 'sccache')
    elif option.name == 'local-rust':
        for path in os.environ['PATH'].split(os.pathsep):
            if os.path.exists(path + '/rustc'):
                set('build.rustc', path + '/rustc')
                break
        for path in os.environ['PATH'].split(os.pathsep):
            if os.path.exists(path + '/cargo'):
                set('build.cargo', path + '/cargo')
                break
    elif option.name == 'local-rust-root':
        set('build.rustc', value + '/bin/rustc')
        set('build.cargo', value + '/bin/cargo')
    elif option.name == 'llvm-root':
        set('target.{}.llvm-config'.format(build()), value + '/bin/llvm-config')
    elif option.name == 'llvm-config':
        set('target.{}.llvm-config'.format(build()), value)
    elif option.name == 'llvm-filecheck':
        set('target.{}.llvm-filecheck'.format(build()), value)
    elif option.name == 'tools':
        set('build.tools', value.split(','))
    elif option.name == 'codegen-backends':
        set('rust.codegen-backends', value.split(','))
    elif option.name == 'host':
        set('build.host', value.split(','))
    elif option.name == 'target':
        set('build.target', value.split(','))
    elif option.name == 'full-tools':
        set('rust.codegen-backends', ['llvm'])
        set('rust.lld', True)
        set('rust.llvm-tools', True)
        set('build.extended', True)
    elif option.name == 'option-checking':
        # this was handled above
        pass
    elif option.name == 'dist-compression-formats':
        set('dist.compression-formats', value.split(','))
    else:
        raise RuntimeError("unhandled option {}".format(option.name))

set('build.configure-args', sys.argv[1:])

# "Parse" the `config.example.toml` file into the various sections, and we'll
# use this as a template of a `config.toml` to write out which preserves
# all the various comments and whatnot.
#
# Note that the `target` section is handled separately as we'll duplicate it
# per configured target, so there's a bit of special handling for that here.
sections = {}
cur_section = None
sections[None] = []
section_order = [None]
targets = {}
top_level_keys = []

for line in open(rust_dir + '/config.example.toml').read().split("\n"):
    if cur_section == None:
        if line.count('=') == 1:
            top_level_key = line.split('=')[0]
            top_level_key = top_level_key.strip(' #')
            top_level_keys.append(top_level_key)
    if line.startswith('['):
        cur_section = line[1:-1]
        if cur_section.startswith('target'):
            cur_section = 'target'
        elif '.' in cur_section:
            raise RuntimeError("don't know how to deal with section: {}".format(cur_section))
        sections[cur_section] = [line]
        section_order.append(cur_section)
    else:
        sections[cur_section].append(line)

# Fill out the `targets` array by giving all configured targets a copy of the
# `target` section we just loaded from the example config
configured_targets = [build()]
if 'build' in config:
    if 'host' in config['build']:
        configured_targets += config['build']['host']
    if 'target' in config['build']:
        configured_targets += config['build']['target']
if 'target' in config:
    for target in config['target']:
        configured_targets.append(target)
for target in configured_targets:
    targets[target] = sections['target'][:]
    # For `.` to be valid TOML, it needs to be quoted. But `bootstrap.py` doesn't use a proper TOML parser and fails to parse the target.
    # Avoid using quotes unless it's necessary.
    targets[target][0] = targets[target][0].replace("x86_64-unknown-linux-gnu", "'{}'".format(target) if "." in target else target)


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
        return '[' + ', '.join(map(to_toml, value)) + ']'
    elif isinstance(value, str):
        # Don't put quotes around numeric values
        if is_number(value):
            return value
        else:
            return "'" + value + "'"
    elif isinstance(value, dict):
        return "{" + ", ".join(map(lambda a: "{} = {}".format(to_toml(a[0]), to_toml(a[1])), value.items())) + "}"
    else:
        raise RuntimeError('no toml')


def configure_section(lines, config):
    for key in config:
        value = config[key]
        found = False
        for i, line in enumerate(lines):
            if not line.startswith('#' + key + ' = '):
                continue
            found = True
            lines[i] = "{} = {}".format(key, to_toml(value))
            break
        if not found:
            # These are used by rpm, but aren't accepted by x.py.
            # Give a warning that they're ignored, but not a hard error.
            if key in ["infodir", "localstatedir"]:
                print("warning: {} will be ignored".format(key))
            else:
                raise RuntimeError("failed to find config line for {}".format(key))


def configure_top_level_key(lines, top_level_key, value):
    for i, line in enumerate(lines):
        if line.startswith('#' + top_level_key + ' = ') or line.startswith(top_level_key + ' = '):
            lines[i] = "{} = {}".format(top_level_key, value)
            return

    raise RuntimeError("failed to find config line for {}".format(top_level_key))


for section_key, section_config in config.items():
    if section_key not in sections and section_key not in top_level_keys:
        raise RuntimeError("config key {} not in sections or top_level_keys".format(section_key))
    if section_key in top_level_keys:
        configure_top_level_key(sections[None], section_key, section_config)

    elif  section_key == 'target':
        for target in section_config:
            configure_section(targets[target], section_config[target])
    else:
        configure_section(sections[section_key], section_config)

def write_uncommented(target, f):
    block = []
    is_comment = True

    for line in target:
        block.append(line)
        if len(line) == 0:
            if not is_comment:
                for l in block:
                    f.write(l + "\n")
            block = []
            is_comment = True
            continue
        is_comment = is_comment and line.startswith('#')
    return f

# Now that we've built up our `config.toml`, write it all out in the same
# order that we read it in.
p("")
p("writing `config.toml` in current directory")
with bootstrap.output('config.toml') as f:
    for section in section_order:
        if section == 'target':
            for target in targets:
                f = write_uncommented(targets[target], f)
        else:
            f = write_uncommented(sections[section], f)

with bootstrap.output('Makefile') as f:
    contents = os.path.join(rust_dir, 'src', 'bootstrap', 'mk', 'Makefile.in')
    contents = open(contents).read()
    contents = contents.replace("$(CFG_SRC_DIR)", rust_dir + '/')
    contents = contents.replace("$(CFG_PYTHON)", sys.executable)
    f.write(contents)

p("")
p("run `python {}/x.py --help`".format(rust_dir))
