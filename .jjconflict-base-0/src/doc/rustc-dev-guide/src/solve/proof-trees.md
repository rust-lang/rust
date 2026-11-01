# Proof trees

While the trait solver itself only returns whether a goal holds and the necessary
constraints, we sometimes also want to know what happened while trying to prove
it. While the trait solver should generally be treated as a black box by the rest
of the compiler, we cannot completely ignore its internals and provide "proof trees"
as an interface for this. To use them you implement the [`ProofTreeVisitor`] trait,
see its existing implementations for examples. The most notable uses are to compute
the [intercrate ambiguity causes for coherence errors][intercrate-ambig],
[improving trait solver errors][solver-errors], and
[eagerly inferring closure signatures][closure-sig].

## Computing proof trees

The trait solver uses [Canonicalization] and uses completely separate `InferCtxt` for
each nested goal. Both diagnostics and auto-traits in rustdoc need to correctly
handle "looking into nested goals". Given a goal like `Vec<Vec<?x>>: Debug`, we
canonicalize to `exists<T0> Vec<Vec<T0>>: Debug`, instantiate that goal as
`Vec<Vec<?0>>: Debug`, get a nested goal `Vec<?0>: Debug`, canonicalize this to get
`exists<T0> Vec<T0>: Debug`, instantiate this as `Vec<?0>: Debug` which then results
in a nested `?0: Debug` goal which is ambiguous.

We compute proof trees by passing a [`ProofTreeBuilder`] to the search graph which is
converting the evaluation steps of the trait solver into a tree. When storing any
data using inference variables or placeholders, the data is canonicalized together
with the list of all unconstrained inference variables created during this computation.
This [`CanonicalState`] is then instantiated in the parent inference context while
walking the proof tree, using the list of inference variables to connect all the
canonicalized values created during this evaluation.

## Debugging the solver

We previously also tried to use proof trees to debug the solver implementation. This
has different design requirements than analyzing it programmatically. The recommended
way to debug the trait solver is by using `tracing`. The trait solver only uses the
`debug` tracing level for its general 'shape' and `trace` for additional detail.
`RUSTC_LOG=rustc_next_trait_solver=debug` therefore gives you a general outline
and `RUSTC_LOG=rustc_next_trait_solver=trace` can then be used if more precise
information is required.

[`ProofTreeVisitor`]: https://github.com/rust-lang/rust/blob/d6c8169c186ab16a3404cd0d0866674018e8a19e/compiler/rustc_trait_selection/src/solve/inspect/analyse.rs#L403
[`ProofTreeBuilder`]: https://github.com/rust-lang/rust/blob/d6c8169c186ab16a3404cd0d0866674018e8a19e/compiler/rustc_next_trait_solver/src/solve/inspect/build.rs#L40
[`CanonicalState`]: https://github.com/rust-lang/rust/blob/d6c8169c186ab16a3404cd0d0866674018e8a19e/compiler/rustc_type_ir/src/solve/inspect.rs#L31-L47
[intercrate-ambig]: https://github.com/rust-lang/rust/blob/d6c8169c186ab16a3404cd0d0866674018e8a19e/compiler/rustc_trait_selection/src/traits/coherence.rs#L742-L748
[solver-errors]: https://github.com/rust-lang/rust/blob/d6c8169c186ab16a3404cd0d0866674018e8a19e/compiler/rustc_trait_selection/src/solve/fulfill.rs#L343-L356
[closure-sig]: https://github.com/rust-lang/rust/blob/d6c8169c186ab16a3404cd0d0866674018e8a19e/compiler/rustc_hir_typeck/src/closure.rs#L333-L339
[Canonicalization]: ./canonicalization.md