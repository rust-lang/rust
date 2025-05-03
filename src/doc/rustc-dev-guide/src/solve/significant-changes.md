## Significant changes and quirks

While some of the items below are already mentioned separately, this page tracks the
main changes from the old trait system implementation. This also mentions some ways
in which the solver significantly diverges from an idealized implementation. This
document simplifies and ignores edge cases. It is recommended to add an implicit
"mostly" to each statement.

### Canonicalization

The new solver uses [canonicalization] when evaluating nested goals. In case there
are possibly multiple candidates, each candidate is eagerly canonicalized. We then
attempt to merge their canonical responses. This differs from the old implementation
which does not use canonicalization inside of the trait system.

This has a some major impacts on the design of both solvers. Without using
canonicalization to stash the constraints of candidates, candidate selection has
to discard the constraints of each candidate, only applying the constraints by
reevaluating the candidate after it has been selected: [source][evaluate_stack].
Without canonicalization it is also not possible to cache the inference constraints
from evaluating a goal. This causes the old implementation to have two systems:
*evaluate* and *fulfill*. *Evaluation* is cached, does not apply inference constraints
and is used when selecting candidates. *Fulfillment* applies inference and region
constraints is not cached and applies inference constraints.

By using canonicalization, the new implementation is able to merge *evaluation* and
*fulfillment*, avoiding complexity and subtle differences in behavior. It greatly
simplifies caching and prevents accidentally relying on untracked information.
It allows us to avoid reevaluating candidates after selection and enables us to merge
the responses of multiple candidates. However, canonicalizing goals during evaluation
forces the new implementation to use a fixpoint algorithm when encountering cycles
during trait solving: [source][cycle-fixpoint].

[canonicalization]: ./canonicalization.md
[evaluate_stack]: https://github.com/rust-lang/rust/blob/47dd709bedda8127e8daec33327e0a9d0cdae845/compiler/rustc_trait_selection/src/traits/select/mod.rs#L1232-L1237
[cycle-fixpoint]: https://github.com/rust-lang/rust/blob/df8ac8f1d74cffb96a93ae702d16e224f5b9ee8c/compiler/rustc_trait_selection/src/solve/search_graph.rs#L382-L387

### Deferred alias equality

The new implementation emits `AliasRelate` goals when relating aliases while the
old implementation structurally relates the aliases instead. This enables the
new solver to stall equality until it is able to normalize the related aliases.

The behavior of the old solver is incomplete and relies on eager normalization
which replaces ambiguous aliases with inference variables. As this is
not possible for aliases containing bound variables, the old implementation does
not handle aliases inside of binders correctly, e.g. [#102048]. See the chapter on
[normalization] for more details.

[#102048]: https://github.com/rust-lang/rust/issues/102048

### Eagerly evaluating nested goals

The new implementation eagerly handles nested goals instead of returning
them to the caller. The old implementation does both. In evaluation nested
goals [are eagerly handled][eval-nested], while fulfillment simply
[returns them for later processing][fulfill-nested].

As the new implementation has to be able to eagerly handle nested goals for
candidate selection, always doing so reduces complexity. It may also enable
us to merge more candidates in the future.

[eval-nested]: https://github.com/rust-lang/rust/blob/master/compiler/rustc_trait_selection/src/traits/select/mod.rs#L1271-L1277
[fulfill-nested]: https://github.com/rust-lang/rust/blob/df8ac8f1d74cffb96a93ae702d16e224f5b9ee8c/compiler/rustc_trait_selection/src/traits/fulfill.rs#L708-L712

### Nested goals are evaluated until reaching a fixpoint

The new implementation always evaluates goals in a loop until reaching a fixpoint.
The old implementation only does so in *fulfillment*, but not in *evaluation*.
Always doing so strengthens inference and is reduces the order dependence of
the trait solver. See [trait-system-refactor-initiative#102].

[trait-system-refactor-initiative#102]: https://github.com/rust-lang/trait-system-refactor-initiative/issues/102

### Proof trees and providing diagnostics information

The new implementation does not track diagnostics information directly,
instead providing [proof trees][trees] which are used to lazily compute the
relevant information. This is not yet fully fleshed out and somewhat hacky.
The goal is to avoid tracking this information in the happy path to improve
performance and to avoid accidentally relying on diagnostics data for behavior.

[trees]: ./proof-trees.md

## Major quirks of the new implementation

### Hiding impls if there are any env candidates

If there is at least one `ParamEnv` or `AliasBound` candidate to prove
some `Trait` goal, we discard all impl candidates for both `Trait` and
`Projection` goals: [source][discard-from-env]. This prevents users from
using an impl which is entirely covered by a `where`-bound,  matching the
behavior of the old implementation and avoiding some weird errors,
e.g. [trait-system-refactor-initiative#76].

[discard-from-env]: https://github.com/rust-lang/rust/blob/03994e498df79aa1f97f7bbcfd52d57c8e865049/compiler/rustc_trait_selection/src/solve/assembly/mod.rs#L785-L789
[trait-system-refactor-initiative#76]: https://github.com/rust-lang/trait-system-refactor-initiative/issues/76

### `NormalizesTo` goals are a function

See the [normalization] chapter. We replace the expected term with an unconstrained
inference variable before computing `NormalizesTo` goals to prevent it from affecting
normalization. This means that `NormalizesTo` goals are handled somewhat differently
from all other goal kinds and need some additional solver support. Most notably,
their ambiguous nested goals are returned to the caller which then evaluates them.
See [#122687] for more details.

[#122687]: https://github.com/rust-lang/rust/pull/122687
[normalization]: ../normalization.md
