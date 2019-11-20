# MIR optimizations

MIR optimizations are optimizations run on the [MIR][mir] to produce better MIR
before codegen. This is important for two reasons: first, it makes the final
generated executable code better, and second, it means that LLVM has less work
to do, so compilation is faster. Note that since MIR is generic (not
[monomorphized][monomorph] yet), these optimizations are particularly
effective; we can optimize the generic version, so all of the monomorphizations
are cheaper!

[mir]: https://rust-lang.github.io/rustc-guide/mir/index.html
[monomorph]: https://rust-lang.github.io/rustc-guide/appendix/glossary.html?highlight=monomorphize#appendix-c-glossary

MIR optimizations run after borrow checking. We run a series of optimization
passes over the MIR to improve it. Some passes are required to run on all code,
some passes don't actually do optimizations but only check stuff, and some
passes are only turned on in `release` mode.

The [`optimized_mir`][optmir] [query] is called to produce the optimized MIR
for a given [`DefId`][defid]. This query makes sure that the borrow checker has
run and that some validation has occurred. Then, it [steals][steal] the MIR
optimizes it, and returns the improved MIR.

[optmir]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/transform/fn.optimized_mir.html
[query]: https://rust-lang.github.io/rustc-guide/query.html
[defid]: https://rust-lang.github.io/rustc-guide/appendix/glossary.html?highlight=DefId#appendix-c-glossary
[steal]: https://rust-lang.github.io/rustc-guide/mir/passes.html?highlight=steal#stealing

## Defining optimization passes

The list of passes run and the order in which they are run is defined by the
[`run_optimization_passes`][rop] function. It contains an array of passes to
run.  Each pass in the array is a struct that implements the [`MirPass`] trait.
The array is an array of `&dyn MirPass` trait objects. Typically, a pass is
implemented in its own submodule of the [`rustc_mir::transform`][trans] module.

[rop]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/transform/fn.run_optimization_passes.html
[`MirPass`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/transform/trait.MirPass.html
[trans]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/transform/index.html 

Some examples of passes are:
- `CleanupNonCodegenStatements`: remove some of the info that is only needed for
  analyses, rather than codegen.
- `ConstProp`: Does [constant propagation][constprop]

You can see the ["Implementors" section of the `MirPass` rustdocs][impl] for more examples.

[impl]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_mir/transform/trait.MirPass.html#implementors
[constprop]: https://en.wikipedia.org/wiki/Constant_folding#Constant_propagation
