# Normalization in the new solver

> **NOTE**: FIXME: The content of this chapter has some been changed quite
significantly since it was written.

With the new solver we've made some fairly significant changes to normalization when compared
to the existing implementation.

We now differentiate between "one-step normalization", "structural normalization" and
"deep normalization".

## One-step normalization

One-step normalization is implemented via `NormalizesTo` goals. Unlike other goals
in the trait solver, `NormalizesTo` always expects the term to be an unconstrained
inference variable[^opaques]. Think of it as a function, taking an alias as input
and returning its underlying value. If the alias is rigid, `NormalizesTo` fails and
returns `NoSolution`. This is the case for `<T as Trait>::Assoc` if there's a `T: Trait`
where-bound and for opaque types with `Reveal::UserFacing` unless they are in the
defining scope. We must not treat any aliases as rigid in coherence.

The underlying value may itself be an unnormalized alias, e.g.
`NormalizesTo(<<() as Id>::This as Id>::This)` only returns `<() as Id>::This`,
even though that alias can be further normalized to `()`. As the term is
always an unconstrained inference variable, the expected term cannot influence
normalization, see [trait-system-refactor-initiative#22] for more.

Only ever computing `NormalizesTo` goals with an unconstrained inference variable
requires special solver support. It is only used by `AliasRelate` goals and pending
`NormalizesTo` goals are tracked separately from other goals: [source][try-eval-norm].
As the expected term is always erased in `NormalizesTo`, we have to return its
ambiguous nested goals to its caller as not doing so weakens inference. See
[#122687] for more details.  

[trait-system-refactor-initiative#22]: https://github.com/rust-lang/trait-system-refactor-initiative/issues/22
[try-eval-norm]: https://github.com/rust-lang/rust/blob/2627e9f3012a97d3136b3e11bf6bd0853c38a534/compiler/rustc_trait_selection/src/solve/eval_ctxt/mod.rs#L523-L537
[#122687]: https://github.com/rust-lang/rust/pull/122687

## `AliasRelate` and structural normalization

We structurally normalize an alias by applying one-step normalization until
we end up with a rigid alias, ambiguity, or overflow. This is done by repeatedly
evaluating `NormalizesTo` goals inside of a snapshot: [source][structural_norm].

`AliasRelate(lhs, rhs)` is implemented by first structurally normalizing both the
`lhs` and the `rhs` and then relating the resulting rigid types (or inference
variables). Importantly, if `lhs` or `rhs` ends up as an alias, this alias can
now be treated as rigid and gets unified without emitting a nested `AliasRelate`
goal: [source][structural-relate].

This means that `AliasRelate` with an unconstrained `rhs` ends up functioning
similar to `NormalizesTo`, acting as a function which fully normalizes `lhs`
before assigning the resulting rigid type to an inference variable. This is used by
`fn structurally_normalize_ty` both [inside] and [outside] of the trait solver.
This has to be used whenever we match on the value of some type, both inside
and outside of the trait solver.

<!--
FIXME: structure, maybe we should have an "alias handling" chapter instead as
talking about normalization without explaining that doesn't make too much
sense.

FIXME: it is likely that this will subtly change again by mostly moving structural
normalization into `NormalizesTo`.
-->

[structural_norm]: https://github.com/rust-lang/rust/blob/2627e9f3012a97d3136b3e11bf6bd0853c38a534/compiler/rustc_trait_selection/src/solve/alias_relate.rs#L140-L175
[structural-relate]: https://github.com/rust-lang/rust/blob/a0569fa8f91b5271e92d2f73fd252de7d3d05b9c/compiler/rustc_trait_selection/src/solve/alias_relate.rs#L88-L107
[inside]: https://github.com/rust-lang/rust/blob/a0569fa8f91b5271e92d2f73fd252de7d3d05b9c/compiler/rustc_trait_selection/src/solve/mod.rs#L278-L299
[outside]: https://github.com/rust-lang/rust/blob/a0569fa8f91b5271e92d2f73fd252de7d3d05b9c/compiler/rustc_trait_selection/src/traits/structural_normalize.rs#L17-L48

## Deep normalization

By walking over a type, and using `fn structurally_normalize_ty` for each encountered
alias, it is possible to deeply normalize a type, normalizing all aliases as much as
possible. However, this only works for aliases referencing bound variables if they are
not ambiguous as we're unable to replace the alias with a corresponding inference
variable without leaking universes.

<!--
FIXME: we previously had to also be careful about instantiating the new inference
variable with another normalizeable alias. Due to our recent changes to generalization,
this should not be the case anymore. Equating an inference variable with an alias
now always uses `AliasRelate` to fully normalize the alias before instantiating the
inference variable: [source][generalize-no-alias]
-->

[generalize-no-alias]: https://github.com/rust-lang/rust/blob/a0569fa8f91b5271e92d2f73fd252de7d3d05b9c/compiler/rustc_infer/src/infer/relate/generalize.rs#L353-L358

## Outside of the trait solver

The core type system - relating types and trait solving - will not need deep
normalization with the new solver. There are still some areas which depend on it.
For these areas there is the function `At::deeply_normalize`. Without additional
trait solver support deep normalization does not always work in case of ambiguity.
Luckily deep normalization is currently only necessary in places where there is no ambiguity.
`At::deeply_normalize` immediately fails if there's ambiguity.

If we only care about the outermost layer of types, we instead use
`At::structurally_normalize` or `FnCtxt::(try_)structurally_resolve_type`.
Unlike `At::deeply_normalize`, structural normalization is also used in cases where we
have to handle ambiguity.

Because this may result in behavior changes depending on how the trait solver handles
ambiguity, it is safer to also require full normalization there. This happens in
`FnCtxt::structurally_resolve_type` which always emits a hard error if the self type ends
up as an inference variable. There are some existing places which have a fallback for
inference variables instead. These places use `try_structurally_resolve_type` instead.

## Why deep normalization with ambiguity is hard

Fully correct deep normalization is very challenging, especially with the new solver 
given that we do not want to deeply normalize inside of the solver. Mostly deeply normalizing
but sometimes failing to do so is bound to cause very hard to minimize and understand bugs.
If possible, avoiding any reliance on deep normalization entirely therefore feels preferable.

If the solver itself does not deeply normalize, any inference constraints returned by the
solver would require normalization. Handling this correctly is ugly. This also means that
we change goals we provide to the trait solver by "normalizing away" some projections.

The way we (mostly) guarantee deep normalization with the old solver is by eagerly replacing
the projection with an inference variable and emitting a nested `Projection` goal. This works
as `Projection` goals in the old solver deeply normalize. Unless we add another `PredicateKind`
for deep normalization to the new solver we cannot emulate this behavior. This does not work
for projections with bound variables, sometimes leaving them unnormalized. An approach which
also supports projections with bound variables will be even more involved. 

[^opaques]: opaque types are currently handled a bit differently. this may change in the future
