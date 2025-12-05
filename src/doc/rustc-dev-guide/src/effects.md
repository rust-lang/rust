# Effects and const condition checking

## The `HostEffect` predicate

[`HostEffectPredicate`]s are a kind of predicate from `~const Tr` or `const Tr`
bounds. It has a trait reference, and a `constness` which could be `Maybe` or
`Const` depending on the bound. Because `~const Tr`, or rather `Maybe` bounds
apply differently based on whichever contexts they are in, they have different
behavior than normal bounds. Where normal trait bounds on a function such as
`T: Tr` are collected within the [`predicates_of`] query to be proven when a
function is called and to be assumed within the function, bounds such as
`T: ~const Tr` will behave as a normal trait bound and add `T: Tr` to the result
from `predicates_of`, but also adds a `HostEffectPredicate` to the
[`const_conditions`] query.

On the other hand, `T: const Tr` bounds do not change meaning across contexts,
therefore they will result in `HostEffect(T: Tr, const)` being added to
`predicates_of`, and not `const_conditions`.

[`HostEffectPredicate`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_type_ir/predicate/struct.HostEffectPredicate.html
[`predicates_of`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.TyCtxt.html#method.predicates_of
[`const_conditions`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.TyCtxt.html#method.const_conditions

## The `const_conditions` query

`predicates_of` represents a set of predicates that need to be proven to use an
item. For example, to use `foo` in the example below:

```rust
fn foo<T>() where T: Default {}
```

We must be able to prove that `T` implements `Default`. In a similar vein,
`const_conditions` represents a set of predicates that need to be proven to use
an item *in const contexts*. If we adjust the example above to use `const` trait
bounds:

```rust
const fn foo<T>() where T: ~const Default {}
```

Then `foo` would get a `HostEffect(T: Default, maybe)` in the `const_conditions`
query, suggesting that in order to call `foo` from const contexts, one must
prove that `T` has a const implementation of `Default`.

## Enforcement of `const_conditions`

`const_conditions` are currently checked in various places. 

Every call in HIR from a const context (which includes `const fn` and `const`
items) will check that `const_conditions` of the function we are calling hold.
This is done in [`FnCtxt::enforce_context_effects`]. Note that we don't check
if the function is only referred to but not called, as the following code needs
to compile:

```rust
const fn hi<T: ~const Default>() -> T {
    T::default()
}
const X: fn() -> u32 = hi::<u32>;
```

For a trait `impl` to be well-formed, we must be able to prove the
`const_conditions` of the trait from the `impl`'s environment. This is checked
in [`wfcheck::check_impl`].

Here's an example:

```rust
const trait Bar {}
const trait Foo: ~const Bar {}
// `const_conditions` contains `HostEffect(Self: Bar, maybe)`

impl const Bar for () {}
impl const Foo for () {}
// ^ here we check `const_conditions` for the impl to be well-formed
```

Methods of trait impls must not have stricter bounds than the method of the
trait that they are implementing. To check that the methods are compatible, a
hybrid environment is constructed with the predicates of the `impl` plus the
predicates of the trait method, and we attempt to prove the predicates of the
impl method. We do the same for `const_conditions`:

```rust
const trait Foo {
    fn hi<T: ~const Default>();
}

impl<T: ~const Clone> Foo for Vec<T> {
    fn hi<T: ~const PartialEq>();
    // ^ we can't prove `T: ~const PartialEq` given `T: ~const Clone` and
    // `T: ~const Default`, therefore we know that the method on the impl
    // is stricter than the method on the trait.
}
```

These checks are done in [`compare_method_predicate_entailment`]. A similar
function that does the same check for associated types is called
[`compare_type_predicate_entailment`]. Both of these need to consider
`const_conditions` when in const contexts.

In MIR, as part of const checking, `const_conditions` of items that are called
are revalidated again in [`Checker::revalidate_conditional_constness`].

[`compare_method_predicate_entailment`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_analysis/check/compare_impl_item/fn.compare_method_predicate_entailment.html
[`compare_type_predicate_entailment`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_analysis/check/compare_impl_item/fn.compare_type_predicate_entailment.html
[`FnCtxt::enforce_context_effects`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_typeck/fn_ctxt/struct.FnCtxt.html#method.enforce_context_effects
[`wfcheck::check_impl`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_analysis/check/wfcheck/fn.check_impl.html
[`Checker::revalidate_conditional_constness`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_const_eval/check_consts/check/struct.Checker.html#method.revalidate_conditional_constness

## `explicit_implied_const_bounds` on associated types and traits

Bounds on associated types, opaque types, and supertraits such as
```rust
trait Foo: ~const PartialEq {
    type X: ~const PartialEq;
}

fn foo() -> impl ~const PartialEq {
    // ^ unimplemented syntax
}
```

Have their bounds represented differently. Unlike `const_conditions` which need
to be proved for callers, and can be assumed inside the definition (e.g. trait
bounds on functions), these bounds need to be proved at definition (at the impl,
or when returning the opaque) but can be assumed for callers. The non-const
equivalent of these bounds are called [`explicit_item_bounds`].

These bounds are checked in [`compare_impl_item::check_type_bounds`] for HIR
typeck, [`evaluate_host_effect_from_item_bounds`] in the old solver and
[`consider_additional_alias_assumptions`] in the new solver.

[`explicit_item_bounds`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.TyCtxt.html#method.explicit_item_bounds
[`compare_impl_item::check_type_bounds`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_hir_analysis/check/compare_impl_item/fn.check_type_bounds.html
[`evaluate_host_effect_from_item_bounds`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_trait_selection/traits/effects/fn.evaluate_host_effect_from_item_bounds.html
[`consider_additional_alias_assumptions`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_next_trait_solver/solve/assembly/trait.GoalKind.html#tymethod.consider_additional_alias_assumptions

## Proving `HostEffectPredicate`s

`HostEffectPredicate`s are implemented both in the [old solver] and the [new
trait solver]. In general, we can prove a `HostEffect` predicate when either of
these conditions are met:

* The predicate can be assumed from caller bounds;
* The type has a `const` `impl` for the trait, *and* that const conditions on
the impl holds, *and* that the `explicit_implied_const_bounds` on the trait
holds; or
* The type has a built-in implementation for the trait in const contexts. For
example, `Fn` may be implemented by function items if their const conditions
are satisfied, or `Destruct` is implemented in const contexts if the type can
be dropped at compile time.

[old solver]: https://doc.rust-lang.org/nightly/nightly-rustc/src/rustc_trait_selection/traits/effects.rs.html
[new trait solver]: https://doc.rust-lang.org/nightly/nightly-rustc/src/rustc_next_trait_solver/solve/effect_goals.rs.html
