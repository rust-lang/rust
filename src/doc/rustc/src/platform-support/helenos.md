# `*-unknown-helenos`

**Tier: 3**

Targets for [HelenOS](https://www.helenos.org).
These targets allow compiling user-space applications, that you can then copy into your HelenOS ISO image to run them.

Target triplets available so far:

- `x86_64-unknown-helenos`
- `i686-unknown-helenos`

## Target maintainers

- Matěj Volf ([@mvolfik](https://github.com/mvolfik))

## Requirements

These targets only support cross-compilation. The targets will[^1] support libstd, although support of some platform features (filesystem, networking) may be limited.

You need to have a local clone of the HelenOS repository and the HelenOS toolchain set up, no HelenOS-Rust development artifacts are available.

[^1]: libstd is not yet available, because it needs to be done in a separate PR, because compiler support needs to be merged first to allow creating libc bindings

## Building

### HelenOS toolchain setup

For compilation of standard library, you need to build the HelenOS toolchain (because Rust needs to use `*-helenos-gcc` as linker) and shared libraries. See [this HelenOS wiki page](https://www.helenos.org/wiki/UsersGuide/CompilingFromSource#a2.Buildasupportedcross-compiler) for instruction on setting up the build. At the end of step 4 (_Configure and build_), invoke `ninja export-dev` to build the shared libraries.

Then copy these shared libraries from `export-dev/lib` to the path where the compiler automatically searches for them. This will be the directory where you installed the toolchain (for example `~/.local/share/HelenOS/cross/i686-helenos/lib`). You can see this path with this command:

```sh
touch /tmp/test.c
i686-helenos-gcc -v -c /tmp/test.c 2>&1 | grep LIBRARY_PATH
```

### Building the target

When you have the HelenOS toolchain set up and installed in your path, you can build the Rust toolchain using the standard procedure. See [rustc dev guide](https://rustc-dev-guide.rust-lang.org/building/how-to-build-and-run.html).

### Building Rust programs

Use the toolchain that you have built above and run `cargo build --target <arch>-unknown-helenos`.

## Testing

After you build a Rust program for HelenOS, you can put it into the `dist` directory of the HelenOS build, build the ISO image, and then run it either in an emulator, or on real hardware. See HelenOS wiki for further instructions on running the OS.

Running the Rust testsuite has not been attempted yet due to missing host tools and networking code.

## Cross-compilation toolchains and C code

You should be able to cross-compile and link any needed C code using `<arch>-helenos-gcc` that you built above. However, note that clang support is highly lacking. Therefore, to run tools such as `bindgen`, you will need to provide flag `-nostdinc` and manually specify the include paths to HelenOS headers, which you will find in the `export-dev` folder + in the cross-compilation toolchain (e.g. `~/.local/share/HelenOS/cross/lib/gcc/i686-helenos/14.2.0/include`).
