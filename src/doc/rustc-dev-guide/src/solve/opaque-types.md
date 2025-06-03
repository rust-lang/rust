# Opaque types in the new solver

The way [opaque types] are handled in the new solver differs from the old implementation.
This should be a self-contained explanation of the behavior in the new solver.

[opaque types]: ../opaque-types-type-alias-impl-trait.md

## opaques are alias types

Opaque types are treated the same as other aliases, most notabily associated types,
whenever possible. There should be as few divergences in behavior as possible.

This is desirable, as they are very similar to other alias types, in that they can be
normalized to their hidden type and also have the same requirements for completeness.
Treating them this way also reduces the complexity of the type system by sharing code.
Having to deal with opaque types separately results in more complex rules and new kinds
of interactions. As we need to treat them like other aliases in the implicit-negative
mode, having significant differences between modes also adds complexity.

*open question: is there an alternative approach here, maybe by treating them more like rigid
types with more limited places to instantiate them? they would still have to be ordinary
aliases during coherence*

### `normalizes-to` for opaques

[source][norm]

`normalizes-to` is used to define the one-step normalization behavior for aliases in the new
solver: `<<T as IdInner>::Assoc as IdOuter>::Assoc` first normalizes to `<T as IdInner>::Assoc`
which then normalizes to `T`. It takes both the `AliasTy` which is getting normalized and the
expected `Term`. To use `normalizes-to` for actual normalization, the expected term can simply
be an unconstrained inference variable.

For opaque types in the defining scope and in the implicit-negative coherence mode, this is
always done in two steps. Outside of the defining scope `normalizes-to` for opaques always
returns `Err(NoSolution)`.

We start by trying to assign the expected type as a hidden type.

In the implicit-negative coherence mode, this currently always results in ambiguity without
interacting with the opaque types storage. We could instead add allow 'defining' all opaque types,
discarding their inferred types at the end, changing the behavior of an opaque type is used
multiple times during coherence: [example][coherence-example]

Inside of the defining scope we start by checking whether the type and const arguments of the
opaque are all placeholders: [source][placeholder-ck]. If this check is ambiguous,
return ambiguity, if it fails, return `Err(NoSolution)`. This check ignores regions which are
only checked at the end of borrowck. If it succeeds, continue.

We then check whether we're able to *semantically* unify the generic arguments of the opaque
with the arguments of any opaque type already in the opaque types storage. If so, we unify the
previously stored type with the expected type of this `normalizes-to` call: [source][eq-prev][^1].

If not, we insert the expected type in the opaque types storage: [source][insert-storage][^2]. 
Finally, we check whether the item bounds of the opaque hold for the expected type:
[source][item-bounds-ck].

[norm]: https://github.com/rust-lang/rust/blob/384d26fc7e3bdd7687cc17b2662b091f6017ec2a/compiler/rustc_trait_selection/src/solve/normalizes_to/opaque_types.rs#L13
[coherence-example]: https://github.com/rust-lang/rust/blob/master/tests/ui/type-alias-impl-trait/coherence_different_hidden_ty.rs
[placeholder-ck]: https://github.com/rust-lang/rust/blob/384d26fc7e3bdd7687cc17b2662b091f6017ec2a/compiler/rustc_trait_selection/src/solve/normalizes_to/opaque_types.rs#L33
[check-storage]: https://github.com/rust-lang/rust/blob/384d26fc7e3bdd7687cc17b2662b091f6017ec2a/compiler/rustc_trait_selection/src/solve/normalizes_to/opaque_types.rs#L51-L52
[eq-prev]: https://github.com/rust-lang/rust/blob/384d26fc7e3bdd7687cc17b2662b091f6017ec2a/compiler/rustc_trait_selection/src/solve/normalizes_to/opaque_types.rs#L51-L59
[insert-storage]: https://github.com/rust-lang/rust/blob/384d26fc7e3bdd7687cc17b2662b091f6017ec2a/compiler/rustc_trait_selection/src/solve/normalizes_to/opaque_types.rs#L68
[item-bounds-ck]: https://github.com/rust-lang/rust/blob/384d26fc7e3bdd7687cc17b2662b091f6017ec2a/compiler/rustc_trait_selection/src/solve/normalizes_to/opaque_types.rs#L69-L74

[^1]: FIXME: this should ideally only result in a unique candidate given that we require the args to be placeholders and regions are always inference vars
[^2]: FIXME: why do we check whether the expected type is rigid for this.

### using alias-bounds of normalizable aliases

https://github.com/rust-lang/trait-system-refactor-initiative/issues/77

Using an `AliasBound` candidate for normalizable aliases is generally not possible as an
associated type can have stronger bounds then the resulting type when normalizing via a
`ParamEnv` candidate.

These candidates would change our exact normalization strategy to be user-facing. It is otherwise
pretty much unobservable whether we eagerly normalize. Where we normalize is something we likely
want to change that after removing support for the old solver, so that would be undesirable.

## opaque types can be defined anywhere

Opaque types in their defining-scope can be defined anywhere, whether when simply relating types
or in the trait solver. This removes order dependence and incompleteness. Without this the result
of a goal can differ due to subtle reasons, e.g. whether we try to evaluate a goal using the
opaque before the first defining use of the opaque.

## higher ranked opaque types in their defining scope

These are not supported and trying to define them right now should always error.

FIXME: Because looking up opaque types in the opaque type storage can now unify regions,
we have to eagerly check that the opaque types does not reference placeholders. We otherwise
end up leaking placeholders.

## member constraints

The handling of member constraints does not change in the new solver. See the
[relevant existing chapter][member-constraints] for that.

[member-constraints]: ../borrow_check/region_inference/member_constraints.md

## calling methods on opaque types

FIXME: We need to continue to support calling methods on still unconstrained
opaque types in their defining scope. It's unclear how to best do this.

```rust
use std::future::Future;
use futures::FutureExt;

fn go(i: usize) -> impl Future<Output = ()> + Send + 'static {
    async move {
        if i != 0 {
            // This returns `impl Future<Output = ()>` in its defining scope,
            // we don't know the concrete type of that opaque at this point.
            // Currently treats the opaque as a known type and succeeds, but
            // from the perspective of "easiest to soundly implement", it would
            // be good for this to be ambiguous.
            go(i - 1).boxed().await;
        }
    }
}
```
