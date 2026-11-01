# `*-unknown-helenos`

**Tier: 3**

Targets for [HelenOS](https://www.helenos.org).
These targets allow compiling user-space applications, that you can then copy into your HelenOS ISO image to run them.

Target triplets available:

- `x86_64-unknown-helenos`
- `sparc64-unknown-helenos`
- `powerpc-unknown-helenos`
- `aarch64-unknown-helenos`
- `i686-unknown-helenos`*


On i686, some portions of native HelenOS libraries run into issues due to vector instructions accessing variables from the stack that seems
to be misaligned. It is not clear if this is fault of HelenOS or Rust. Most programs work, but for example calling `ui_window_create` from HelenOS
libui does not work.

## Target maintainers

- Matěj Volf ([@mvolfik](https://github.com/mvolfik))

## Requirements

These targets only support cross-compilation. The targets will[^helenos-libstd-pending] support libstd, although support of some platform features (filesystem, networking) may be limited.

You need to have a local clone of the HelenOS repository and the HelenOS toolchain set up, no HelenOS-Rust development artifacts are available.

[^helenos-libstd-pending]: libstd is not yet available, because it needs to be done in a separate PR, because compiler support needs to be merged first to allow creating libc bindings

## Building

If you want to avoid the full setup, fully automated Docker-based build system is available at https://github.com/mvolfik/helenos-rust-autobuild

### HelenOS toolchain setup

For compilation of standard library, you need to build the HelenOS toolchain (because Rust needs to use `*-helenos-gcc` as linker) and its libraries (libc and a few others). See [this HelenOS wiki page](https://www.helenos.org/wiki/UsersGuide/CompilingFromSource#a2.Buildasupportedcross-compiler) for instruction on setting up the build. At the end of step 4 (_Configure and build_), after `ninja image_path`, invoke `ninja export-dev` to build the shared libraries.

Copy the libraries to the path where the compiler automatically searches for them. This will be the directory where you installed the toolchain (for example `~/.local/share/HelenOS/cross/i686-helenos/lib`). In the folder where you built HelenOS, you can run these commands:

```sh
touch /tmp/test.c
HELENOS_LIB_PATH="$(realpath "$(amd64-helenos-gcc -v -c /tmp/test.c 2>&1 | grep LIBRARY_PATH | cut -d= -f2 | cut -d: -f2)")"
# use sparc64-helenos-gcc above for the SPARC toolchain, etc
cp -P export-dev/lib/* "$HELENOS_LIB_PATH"
```

### Building the target

When you have the HelenOS toolchain set up and installed in your path, you can build the Rust toolchain using the standard procedure. See [rustc dev guide](https://rustc-dev-guide.rust-lang.org/building/how-to-build-and-run.html).

In the most simple case, this means that you can run `./x build library --stage 1 --target x86_64-unknown-linux-gnu,<arch>-unknown-helenos` (the first target triple should be your host machine, adjust accordingly). Then run `rustup toolchain link mytoolchain build/host/stage1` to allow using your toolchain for building Rust programs.

### Building Rust programs

If you linked the toolchain above as `mytoolchain`, run `cargo +mytoolchain build --target <arch>-unknown-helenos`.

## Testing

After you build a Rust program for HelenOS, you can put it into the `dist` directory of the HelenOS build, build the ISO image, and then run it either in an emulator, or on real hardware. See HelenOS wiki for further instructions on running the OS.

Running the Rust testsuite has not been attempted yet due to missing host tools (thus the test suite can't be run natively) and insufficient networking support (thus we can't use the `remote-test-server` tool).

## Cross-compilation toolchains and C code

You should be able to cross-compile and link any needed C code using `<arch>-helenos-gcc` that you built above. However, note that clang support is highly lacking. Therefore, to run tools such as `bindgen`, you will need to provide flag `-nostdinc` and manually specify the include paths to HelenOS headers, which you will find in the `export-dev` folder + in the cross-compilation toolchain (e.g. `~/.local/share/HelenOS/cross/lib/gcc/i686-helenos/14.2.0/include`). You can see an example of proper build.rs at https://github.com/mvolfik/helenos-ui-rs/blob/master/build.rs
