# Building and testing with changes in rustc code

This is useful when changing code in `rustc_codegen_cranelift` as part of changing [main Rust repository](https://github.com/rust-lang/rust/).
This can happen, for example, when you are implementing a new compiler intrinsic.

Instruction below uses `$RustCheckoutDir` as substitute for any folder where you cloned Rust repository.

You need to do this steps to successfully compile and use the cranelift backend with your changes in rustc code:

1. `cd $RustCheckoutDir`
2. Run `python x.py setup` and choose option for compiler (`b`).
3. Build compiler and necessary tools: `python x.py build --stage=2 compiler library/std src/tools/rustdoc src/tools/rustfmt`
   * (Optional) You can also build cargo by adding `src/tools/cargo` to previous command.
4. Copy cargo from a nightly toolchain: `cp $(rustup +nightly which cargo) ./build/host/stage2/bin/cargo`. Note that you would need to do this every time you rebuilt `rust` repository.
5. Link your new `rustc` to toolchain: `rustup toolchain link stage2 ./build/host/stage2/`.
6. (Windows only) compile the build system: `rustc +stage2 -O build_system/main.rs -o y.exe`.
7. You need to prefix every `./y.sh` (or `y` if you built `build_system/main.rs` as `y`) command by `rustup run stage2` to make cg_clif use your local changes in rustc.
  * `rustup run stage2 ./y.sh prepare`
  * `rustup run stage2 ./y.sh build`
  * (Optional) run tests: `rustup run stage2 ./y.sh test`
8. Now you can use your cg_clif build to compile other Rust programs, e.g. you can open any Rust crate and run commands like `$RustCheckoutDir/compiler/rustc_codegen_cranelift/dist/cargo-clif build --release`.

You can also set `rust-analyzer.rustc.source` to your rust workspace to get rust-analyzer to understand your changes.
