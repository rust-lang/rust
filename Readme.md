# Work in progress cranelift codegen backend for rust

> ⚠⚠⚠ This doesn't do much useful yet ⚠⚠⚠

## Building

```bash
$ git clone https://github.com/bjorn3/rustc_codegen_cranelift.git
$ cd rustc_codegen_cranelift
$ ./prepare.sh # download and patch sysroot src and install hyperfine for benchmarking
$ ./test.sh
```

## Usage

`$cg_clif_dir` is the directory you cloned this repo into in the following instruction.

### Rustc

```bash
$ rustc -Cpanic=abort -Zcodegen-backend=$cg_clif_dir/target/debug/librustc_codegen_cranelift.so --sysroot $cg_clif_dir/build_sysroot/sysroot my_crate.rs
```

### Cargo

```bash
$ RUSTFLAGS="-Cpanic=abort -Zcodegen-backend=$cg_clif_dir/target/debug/librustc_codegen_cranelift.dylib --sysroot $cg_clif_dir/build_sysroot/sysroot" cargo run
```

## Not yet supported

* Good non-rust abi support ([vectors are passed by-ref](https://github.com/bjorn3/rustc_codegen_cranelift/issues/10))
* Checked binops ([some missing instructions in cranelift](https://github.com/CraneStation/cranelift/issues/460))
* Inline assembly ([no cranelift support](https://github.com/CraneStation/cranelift/issues/444))
* SIMD ([tracked here](https://github.com/bjorn3/rustc_codegen_cranelift/issues/171))

## Troubleshooting

### Can't compile

Try updating your nightly compiler. You can try to use an nightly a day or two older if updating rustc doesn't fix it. If you still can't compile it, please fill an issue.
