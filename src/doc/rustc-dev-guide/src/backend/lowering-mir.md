# Lowering MIR to a Codegen IR

Now that we have a list of symbols to generate from the collector, we need to
generate some sort of codegen IR. In this chapter, we will assume LLVM IR,
since that's what rustc usually uses. The actual monomorphization is performed
as we go, while we do the translation.

Recall that the backend is started by
[`rustc_codegen_ssa::base::codegen_crate`][codegen1]. Eventually, this reaches
[`rustc_codegen_ssa::mir::codegen_mir`][codegen2], which does the lowering from
MIR to LLVM IR.

[codegen1]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_codegen_ssa/base/fn.codegen_crate.html
[codegen2]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_codegen_ssa/mir/fn.codegen_mir.html

The code is split into modules which handle particular MIR primitives:

- [`rustc_codegen_ssa::mir::block`][mirblk] will deal with translating
  blocks and their terminators.  The most complicated and also the most
  interesting thing this module does is generating code for function calls,
  including the necessary unwinding handling IR.
- [`rustc_codegen_ssa::mir::statement`][mirst] translates MIR statements.
- [`rustc_codegen_ssa::mir::operand`][mirop] translates MIR operands.
- [`rustc_codegen_ssa::mir::place`][mirpl] translates MIR place references.
- [`rustc_codegen_ssa::mir::rvalue`][mirrv] translates MIR r-values.

[mirblk]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_codegen_ssa/mir/block/index.html
[mirst]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_codegen_ssa/mir/statement/index.html
[mirop]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_codegen_ssa/mir/operand/index.html
[mirpl]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_codegen_ssa/mir/place/index.html
[mirrv]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_codegen_ssa/mir/rvalue/index.html

Before a function is translated a number of simple and primitive analysis
passes will run to help us generate simpler and more efficient LLVM IR. An
example of such an analysis pass would be figuring out which variables are
SSA-like, so that we can translate them to SSA directly rather than relying on
LLVM's `mem2reg` for those variables. The analysis can be found in
[`rustc_codegen_ssa::mir::analyze`][mirana].

[mirana]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_codegen_ssa/mir/analyze/index.html

Usually a single MIR basic block will map to a LLVM basic block, with very few
exceptions: intrinsic or function calls and less basic MIR statements like
`assert` can result in multiple basic blocks. This is a perfect lede into the
non-portable LLVM-specific part of the code generation. Intrinsic generation is
fairly easy to understand as it involves very few abstraction levels in between
and can be found in [`rustc_codegen_llvm::intrinsic`][llvmint].

[llvmint]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_codegen_llvm/intrinsic/index.html

Everything else will use the [builder interface][builder]. This is the code that gets
called in the [`rustc_codegen_ssa::mir::*`][ssamir] modules discussed above.

[builder]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_codegen_llvm/builder/index.html
[ssamir]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_codegen_ssa/mir/index.html

> TODO: discuss how constants are generated
