# Having separate `Trait` and `Projection` bounds

Given `T: Foo<AssocA = u32, AssocB = i32>` where-bound, we currently lower it to a `Trait(Foo<T>)` and separate `Projection(<T as Foo>::AssocA, u32)` and `Projection(<T as Foo>::AssocB, i32)` bounds. Why do we not represent this as a single `Trait(Foo[T], [AssocA = u32, AssocB = u32]` bound instead?

The way we prove `Projection` bounds directly relies on proving the corresponding `Trait` bound:
- old solver: https://github.com/rust-lang/rust/blob/461e9738a47e313e4457957fa95ff6a19a4b88d4/compiler/rustc_trait_selection/src/traits/project.rs#L898
- new solver: https://github.com/rust-lang/rust/blob/461e9738a47e313e4457957fa95ff6a19a4b88d4/compiler/rustc_next_trait_solver/src/solve/normalizes_to/mod.rs#L37-L41

We may use a different candidate for norm than for the corresponding trait bound:
- https://rustc-dev-guide.rust-lang.org/solve/candidate-preference.html#we-always-consider-aliasbound-candidates
- https://rustc-dev-guide.rust-lang.org/solve/candidate-preference.html#we-prefer-orphaned-where-bounds

There are also some other subtle reasons for why we can't do so. The most stupid is that for rigid aliases, trying to normalize them does not consider any lifetime constraints from proving the trait bound. This is necessary due to a lack of assumptions on binders - https://github.com/rust-lang/trait-system-refactor-initiative/issues/177 - and should be fixed longterm.

A separate issue is that right now, fetching the `type_of` associated types for `Trait` goals or in shadowed `Projection` candidates can cause query cycles for RPITIT. See https://github.com/rust-lang/trait-system-refactor-initiative/issues/185.

There are also slight differences between candidates for some of the builtin impls, these do all seem generally undesirable and I consider them to be bugs which would be fixed if we had a unified approach here.

Finally, not having this split makes lowering where-clauses more annoying. With the current system having duplicate where-clauses is not an issue and it can easily happen when elaborating super trait bounds. We now need to make sure we merge all associated type constraints, e.g.

```rust
trait Super {
    type A;
    type B;
}

trait Trait: Super<A = i32> {}
// how to elaborate Trait<B = u32>
```
Or even worse
```rust
trait Super<'a> {
    type A;
    type B;
}

trait Trait<'a>: Super<'a, A = i32> {}
// how to elaborate
// T: Trait<'a> + for<'b> Super<'b, B = u32>
```

