# The solver

Also consider reading the documentation for [the recursive solver in chalk][chalk]
as it is very similar to this implementation and also talks about limitations of this
approach.

[chalk]: https://rust-lang.github.io/chalk/book/recursive.html

## A rough walkthrough

The entry-point of the solver is `InferCtxtEvalExt::evaluate_root_goal`. This
function sets up the root `EvalCtxt` and then calls `EvalCtxt::evaluate_goal`,
to actually enter the trait solver.

`EvalCtxt::evaluate_goal` handles [canonicalization](./canonicalization.md), caching,
overflow, and solver cycles. Once that is done, it creates a nested `EvalCtxt` with a
separate local `InferCtxt` and calls `EvalCtxt::compute_goal`, which is responsible for the
'actual solver behavior'. We match on the `PredicateKind`, delegating to a separate function
for each one.

For trait goals, such a `Vec<T>: Clone`, `EvalCtxt::compute_trait_goal` has
to collect all the possible ways this goal can be proven via
`EvalCtxt::assemble_and_evaluate_candidates`. Each candidate is handled in
a separate "probe", to not leak inference constraints to the other candidates.
We then try to merge the assembled candidates via `EvalCtxt::merge_candidates`.


## Important concepts and design pattern

### `EvalCtxt::add_goal`

To prove nested goals, we don't directly call `EvalCtxt::compute_goal`, but instead
add the goal to the `EvalCtxt` with `EvalCtxt::all_goal`. We then prove all nested
goals together in either `EvalCtxt::try_evaluate_added_goals` or
`EvalCtxt::evaluate_added_goals_and_make_canonical_response`. This allows us to handle
inference constraints from later goals.

E.g. if we have both `?x: Debug` and `(): ConstrainToU8<?x>` as nested goals,
then proving `?x: Debug` is initially ambiguous, but after proving `(): ConstrainToU8<?x>`
we constrain `?x` to `u8` and proving `u8: Debug` succeeds.

### Matching on `TyKind`

We lazily normalize types in the solver, so we always have to assume that any types
and constants are potentially unnormalized. This means that matching on `TyKind` can easily
be incorrect.

We handle normalization in two different ways. When proving `Trait` goals when normalizing
associated types, we separately assemble candidates depending on whether they structurally
match the self type. Candidates which match on the self type are handled in
`EvalCtxt::assemble_candidates_via_self_ty` which recurses via
`EvalCtxt::assemble_candidates_after_normalizing_self_ty`, which normalizes the self type
by one level. In all other cases we have to match on a `TyKind` we first use
`EvalCtxt::try_normalize_ty` to normalize the type as much as possible.

### Higher ranked goals

In case the goal is higher-ranked, e.g. `for<'a> F: FnOnce(&'a ())`, `EvalCtxt::compute_goal`
eagerly instantiates `'a` with a placeholder and then recursively proves
`F: FnOnce(&'!a ())` as a nested goal.

### Dealing with choice

Some goals can be proven in multiple ways. In these cases we try each option in
a separate "probe" and then attempt to merge the resulting responses by using
`EvalCtxt::try_merge_responses`. If merging the responses fails, we use
`EvalCtxt::flounder` instead, returning ambiguity. For some goals, we try
incompletely prefer some choices over others in case `EvalCtxt::try_merge_responses`
fails.

## Learning more

The solver should be fairly self-contained. I hope that the above information provides a
good foundation when looking at the code itself. Please reach out on zulip if you get stuck
while doing so or there are some quirks and design decisions which were unclear and deserve
better comments or should be mentioned here.
