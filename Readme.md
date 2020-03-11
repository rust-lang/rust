# WIP Cranelift codegen backend for rust

> ⚠⚠⚠ Threads and certain kinds of FFI don't work yet. ⚠⚠⚠

## Building

```bash
$ git clone https://github.com/bjorn3/rustc_codegen_cranelift.git
$ cd rustc_codegen_cranelift
$ ./prepare.sh # download and patch sysroot src and install hyperfine for benchmarking
$ ./test.sh --release
```

## Usage

`$cg_clif_dir` is the directory you cloned this repo into in the following instruction.

### Cargo

```bash
$ export CG_CLIF_INCR_CACHE=1 # Enable caching of object files in the incremental cache
$ CHANNEL="release" $cg_clif_dir/cargo.sh run
```

If you compiled cg_clif in debug mode you should use `CHANNEL="debug"` instead or omit `CHANNEL="release"` completely.

### Rustc

```bash
$ export CG_CLIF_INCR_CACHE=1 # Enable caching of object files in the incremental cache
$ rustc -Cpanic=abort -Zcodegen-backend=$cg_clif_dir/target/release/librustc_codegen_cranelift.so --sysroot $cg_clif_dir/build_sysroot/sysroot my_crate.rs
```


## Not yet supported

* Good non-rust abi support ([several problems](https://github.com/bjorn3/rustc_codegen_cranelift/issues/10))
* Checked binops ([some missing instructions in cranelift](https://github.com/CraneStation/cranelift/issues/460))
* Inline assembly ([no cranelift support](https://github.com/CraneStation/cranelift/issues/444), not coming soon)
* SIMD ([tracked here](https://github.com/bjorn3/rustc_codegen_cranelift/issues/171), some basic things work)
