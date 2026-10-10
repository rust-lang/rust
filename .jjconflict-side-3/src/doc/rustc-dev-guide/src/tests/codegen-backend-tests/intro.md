# Codegen backend testing

See also the [Code generation](../../backend/codegen.md) chapter.

In addition to the primary LLVM codegen backend, the rust-lang/rust CI also runs tests of the [cranelift][cg_clif] and [GCC][cg_gcc] codegen backends in certain test jobs.

For more details on the tests involved, see:

- [Cranelift codegen backend tests](./cg_clif.md)
- [GCC codegen backend tests](./cg_gcc.md)

[cg_clif]: https://github.com/rust-lang/rustc_codegen_cranelift
[cg_gcc]: https://github.com/rust-lang/rustc_codegen_gcc
