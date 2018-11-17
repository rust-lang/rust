# Work in progress cranelift codegen backend for rust

> ⚠⚠⚠ This doesn't do much useful yet ⚠⚠⚠

## Building

```bash
$ git clone https://github.com/bjorn3/rustc_codegen_cranelift.git
$ cd rustc_codegen_cranelift
$ rustup override set nightly # This uses unstable api's which will never be stabilized
$ cargo install xargo         # Used for building the sysroot
$ cargo install hyperfine     # Used for benchmarking in build.sh
$ cargo build
```

## Usage

```bash
$ rustc -Zcodegen-backend=$(pwd)/target/debug/librustc_codegen_cranelift.so my_crate.rs
```

## Build sysroot and test

```bash
$ rustup component add rust-src # Make sure the sysroot source is available
$ ./prepare_libcore.sh          # Patch the sysroot source for some not yet supported things
$ ./build.sh
```

## Not yet supported

* Good non-rust abi support ([non scalars are not yet supported for the "C" abi](https://github.com/bjorn3/rustc_codegen_cranelift/issues/10))
* Checked binops ([some missing instructions in cranelift](https://github.com/CraneStation/cranelift/issues/460))
* Inline assembly ([no cranelift support](https://github.com/CraneStation/cranelift/issues/444))
* Varargs ([no cranelift support](https://github.com/CraneStation/cranelift/issues/212))
* libstd (needs varargs and some other stuff) ([tracked here](https://github.com/bjorn3/rustc_codegen_cranelift/issues/146))
* u128 and i128 ([no cranelift support](https://github.com/CraneStation/cranelift/issues/354))
* SIMD (huge amount of work to get all intrinsics implemented, so may never be supported)

## Troubleshooting

### Can't compile

Try updating your nightly compiler. You can try to use an nightly a day or two older if updating rustc doesn't fix it. If you still can't compile it, please fill an issue.

### error[E0463]: can't find crate for `std` while compiling a no_std crate

If you use `RUSTFLAGS` to pass `-Zcodegen-backend` to rustc, cargo will compile `build-dependencies` with those flags too. Because this project doesn't support libstd yet, that will result in an error. I don't know of any way to fix this. :(
