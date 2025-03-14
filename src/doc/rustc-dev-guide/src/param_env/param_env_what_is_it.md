
# What is a `ParamEnv`?

The type system relies on information in the environment in order for it to function correctly. This information is stored in the [`ParamEnv`][pe] type and it is important to use the correct `ParamEnv` when interacting with the type system.

The information represented by `ParamEnv` is a list of in-scope where-clauses, and a `Reveal` (see linked docs for more information). A `ParamEnv` typically corresponds to a specific item's where clauses, some clauses are not explicitly written bounds and instead are implicitly added in [`predicates_of`][predicates_of] such as `ConstArgHasType` or some implied bounds.

A `ParamEnv` can also be created with arbitrary data that is not derived from a specific item such as in [`compare_method_predicate_entailment`][method_pred_entailment] which creates a hybrid `ParamEnv` consisting of the impl's where clauses and the trait definition's function's where clauses. In most cases `ParamEnv`s are initially created via the [`param_env` query][query] which returns a `ParamEnv` derived from the provided item's where clauses.

If we have a function such as:
```rust
// `foo` would have a `ParamEnv` of:
// `[T: Sized, T: Trait, <T as Trait>::Assoc: Clone]`
fn foo<T: Trait>()
where
    <T as Trait>::Assoc: Clone,
{}
```
If we were conceptually inside of `foo` (for example, type-checking or linting it) we would use this `ParamEnv` everywhere that we interact with the type system. This would allow things such as normalization (TODO: write a chapter about normalization and link it), evaluating generic constants, and proving where clauses/goals, to rely on `T` being sized, implementing `Trait`, etc.

A more concrete example:
```rust
// `foo` would have a `ParamEnv` of:
// `[T: Sized, T: Clone]`
fn foo<T: Clone>(a: T) {
    // when typechecking `foo` we require all the where clauses on `bar`
    // to hold in order for it to be legal to call. This means we have to
    // prove `T: Clone`. As we are type checking `foo` we use `foo`'s
    // environment when trying to check that `T: Clone` holds.
    //
    // Trying to prove `T: Clone` with a `ParamEnv` of `[T: Sized, T: Clone]`
    // will trivially succeed as bound we want to prove is in our environment.
    requires_clone(a);
}
```

Or alternatively an example that would not compile:
```rust
// `foo2` would have a `ParamEnv` of:
// `[T: Sized]`
fn foo2<T>(a: T) {
    // When typechecking `foo2` we attempt to prove `T: Clone`.
    // As we are type checking `foo2` we use `foo2`'s environment
    // when trying to prove `T: Clone`.
    //
    // Trying to prove `T: Clone` with a `ParamEnv` of `[T: Sized]` will
    // fail as there is nothing in the environment telling the trait solver
    // that `T` implements `Clone` and there exists no user written impl
    // that could apply.
    requires_clone(a);
}
```

It's very important to use the correct `ParamEnv` when interacting with the type system as otherwise it can lead to ICEs or things compiling when they shouldn't (or vice versa). See [#82159](https://github.com/rust-lang/rust/pull/82159) and [#82067](https://github.com/rust-lang/rust/pull/82067) as examples of PRs that changed rustc to use the correct param env to avoid ICE. Determining how to acquire the correct `ParamEnv` is explained later in this chapter.

[predicates_of]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_analysis/collect/predicates_of/fn.predicates_of.html
[method_pred_entailment]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_analysis/check/compare_impl_item/fn.compare_method_predicate_entailment.html
[pe]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.ParamEnv.html
[query]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/context/struct.TyCtxt.html#method.param_env
