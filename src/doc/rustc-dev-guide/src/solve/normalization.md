# Normalization in the new solver

With the new solver we've made some fairly significant changes to normalization when compared
to the existing implementation.

We now differentiate between "shallow normalization" and "deep normalization".
"Shallow normalization" normalizes a type until it is no-longer a potentially normalizeable alias;
it does not recurse into the type. "deep normalization" replaces all normalizeable aliases in a
type with its underlying type.

The old trait solver currently always deeply normalizes via `Projection` obligations.
This is the only way to normalize in the old solver. By replacing projections with a new
inference variable and then emitting `Projection(<T as Trait>::Assoc, ?new_infer)` the old
solver successfully deeply normalizes even in the case of ambiguity. This approach does not
work for projections referencing bound variables.

## Inside of the trait solver

Normalization in the new solver exclusively happens via `Projection`[^0] goals.
This only succeeds by first normalizing the alias by one level and then equating
it with the expected type. This differs from [the behavior of projection clauses]
which can also be proven by successfully equating the projection without normalizating.
This means that `Projection`[^0] goals must only be used in places where we
*have to normalize* to make progress. To normalize `<T as Trait>::Assoc`, we first create
a fresh inference variable `?normalized` and then prove
`Projection(<T as Trait>::Assoc, ?normalized)`[^0]. `?normalized` is then constrained to
the underlying type. 

Inside of the trait solver we never deeply normalize. we only apply shallow normalization
in [`assemble_candidates_after_normalizing_self_ty`] and inside for [`AliasRelate`]
goals for the [`normalizes-to` candidates].

## Outside of the trait solver

The core type system - relating types and trait solving - will not need deep
normalization with the new solver. There are still some areas which depend on it.
For these areas there is the function `At::deeply_normalize`. Without additional
trait solver support deep normalization does not always work in case of ambiguity.
Luckily deep normalization is currently only necessary in places where there is no ambiguity.
`At::deeply_normalize` immediately fails if there's ambiguity.

If we only care about the outermost layer of types, we instead use
`At::structurally_normalize` or `FnCtxt::(try_)structurally_resolve_type`.
Unlike `At::deeply_normalize`, shallow normalization is also used in cases where we
have to handle ambiguity. `At::structurally_normalize` normalizes until the self type
is either rigid or an inference variable and we're stuck with ambiguity. This means
that the self type may not be fully normalized after `At::structurally_normalize` was called.

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


[`assemble_candidates_after_normalizing_self_ty`]: https://github.com/rust-lang/rust/blob/1b6d4cdc4d923c148198ad4df230af48cdaca59e/compiler/rustc_trait_selection/src/solve/assembly/mod.rs#L330-L378
[`AliasRelate`]: https://github.com/rust-lang/rust/blob/1b6d4cdc4d923c148198ad4df230af48cdaca59e/compiler/rustc_trait_selection/src/solve/alias_relate.rs#L16-L102
[`normalizes-to` candidates]: https://github.com/rust-lang/rust/blob/1b6d4cdc4d923c148198ad4df230af48cdaca59e/compiler/rustc_trait_selection/src/solve/alias_relate.rs#L105-L151
[the behavior of projection clauses]: https://github.com/rust-lang/trait-system-refactor-initiative/issues/1
[normalize-via-infer]: https://github.com/rust-lang/rust/blob/1b6d4cdc4d923c148198ad4df230af48cdaca59e/compiler/rustc_trait_selection/src/solve/assembly/mod.rs#L350-L358

[^0]: TODO: currently refactoring this to use `NormalizesTo` predicates instead.