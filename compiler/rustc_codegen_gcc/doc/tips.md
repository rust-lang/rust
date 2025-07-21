# Tips

The following shows how to do different random small things we encountered and thought could
be useful.

### How to send arguments to the GCC linker

```
CG_RUSTFLAGS="-Clink-args=-save-temps -v" ../y.sh cargo build
```

### How to see the personality functions in the asm dump

```
CG_RUSTFLAGS="-Clink-arg=-save-temps -v -Clink-arg=-dA" ../y.sh cargo build
```

### How to see the LLVM IR for a sysroot crate

```
cargo build -v --target x86_64-unknown-linux-gnu -Zbuild-std
# Take the command from the output and add --emit=llvm-ir
```

### To prevent the linker from unmangling symbols

Run with:

```
COLLECT_NO_DEMANGLE=1
```

### How to use a custom-build rustc

 * Build the stage2 compiler (`rustup toolchain link debug-current build/x86_64-unknown-linux-gnu/stage2`).
 * Clean and rebuild the codegen with `debug-current` in the file `rust-toolchain`.

### How to use a custom sysroot source path

If you wish to build a custom sysroot, pass the path of your sysroot source to `--sysroot-source` during the `prepare` step, like so:

```
./y.sh prepare --sysroot-source /path/to/custom/source
```

### How to use [mem-trace](https://github.com/antoyo/mem-trace)

`rustc` needs to be built without `jemalloc` so that `mem-trace` can overload `malloc` since `jemalloc` is linked statically, so a `LD_PRELOAD`-ed library won't a chance to intercept the calls to `malloc`.

### How to generate GIMPLE

If you need to check what gccjit is generating (GIMPLE), then take a look at how to
generate it in [gimple.md](./doc/gimple.md).

### How to build a cross-compiling libgccjit

#### Building libgccjit

 * Follow the instructions on [this repo](https://github.com/cross-cg-gcc-tools/cross-gcc).

#### Configuring rustc_codegen_gcc

 * Run `./y.sh prepare --cross` so that the sysroot is patched for the cross-compiling case.
 * Set the path to the cross-compiling libgccjit in `gcc-path` (in `config.toml`).
 * Make sure you have the linker for your target (for instance `m68k-unknown-linux-gnu-gcc`) in your `$PATH`. You can specify which linker to use via `CG_RUSTFLAGS="-Clinker=<linker>"`, for instance: `CG_RUSTFLAGS="-Clinker=m68k-unknown-linux-gnu-gcc"`. Specify the target when building the sysroot: `./y.sh build --sysroot --target-triple m68k-unknown-linux-gnu`.
 * Build your project by specifying the target and the linker to use: `CG_RUSTFLAGS="-Clinker=m68k-unknown-linux-gnu-gcc" ../y.sh cargo build --target m68k-unknown-linux-gnu`.

If the target is not yet supported by the Rust compiler, create a [target specification file](https://docs.rust-embedded.org/embedonomicon/custom-target.html) (note that the `arch` specified in this file must be supported by the rust compiler).
Then, you can use it the following way:

 * Add the target specification file using `--target` as an **absolute** path to build the sysroot: `./y.sh build --sysroot --target-triple m68k-unknown-linux-gnu --target $(pwd)/m68k-unknown-linux-gnu.json`
 * Build your project by specifying the target specification file: `../y.sh cargo build --target path/to/m68k-unknown-linux-gnu.json`.

If you get the following error:

```
/usr/bin/ld: unrecognised emulation mode: m68kelf
```

Make sure you set `gcc-path` (in `config.toml`) to the install directory.
