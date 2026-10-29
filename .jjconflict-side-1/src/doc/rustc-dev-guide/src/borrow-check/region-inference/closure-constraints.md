# Propagating closure constraints

When we are checking the type tests and universal regions, we may come
across a constraint that we can't prove yet if we are in a closure
body! However, the necessary constraints may actually hold (we just
don't know it yet). Thus, if we are inside a closure, we just collect
all the constraints we can't prove yet and return them. Later, when we
are borrow check the MIR node that created the closure, we can also
check that these constraints hold. At that time, if we can't prove
they hold, we report an error.

## How this is implemented

While borrow-checking a closure inside of `RegionInferenceContext::solve` we separately try to propagate type-outlives and region-outlives constraints to the parent if we're unable to prove them locally.

### Region-outlive constraints

If `RegionInferenceContext::check_universal_regions` fails to prove some outlives constraint `'longer_fr: 'shorter_fr`, we try to propagate it in `fn try_propagate_universal_region_error`. Both these universal regions are either local to the closure or an external region.

In case `'longer_fr` is a local universal region, we search for the largest external region `'fr_minus` which is outlived by `'longer_fr`, i.e. `'longer_fr: 'fr_minus`. In case there are multiple such regions, we pick the `mutual_immediate_postdominator`: the fixpoint of repeatedly computing the GLB of all GLBs, see [TransitiveRelation::postdom_upper_bound](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_data_structures/transitive_relation/struct.TransitiveRelation.html#method.postdom_upper_bound) for more details.

If `'fr_minus` exists we require it to outlive all non-local upper bounds of `'shorter_fr`. There will always be at least one non-local upper bound `'static`.

### Type-outlive constraints

Type-outlives constraints are proven in `check_type_tests`. This happens after computing the outlives graph, which is now immutable.

For all type tests we fail to prove via `fn eval_verify_bound` inside of the closure we call `try_promote_type_test`. A `TypeTest` represents a type-outlives bound `generic_kind: lower_bound` together with a `verify_bound`. If the `VerifyBound` holds for the `lower_bound`, the constraint is satisfied. `try_promote_type_test`  does not care about the ` verify_bound`.

It starts by calling `fn try_promote_type_test_subject`. This function takes the `GenericKind` and tries to transform it to a `ClosureOutlivesSubject`  which is no longer references anything local to the closure. This is done by replacing all free regions in that type with either `'static`  or region parameters which are equal to that free region. This operation fails if the `generic_kind` contains a region which cannot be replaced.

We then promote the `lower_bound` into the context of the caller. If the lower bound is equal to a placeholder, we replace it with `'static`

We then look at all universal regions `uv` which are required to be outlived by `lower_bound`, i.e. for which borrow checking added region constraints. For each of these we then emit a `ClosureOutlivesRequirement` for all non-local universal regions which are known to outlive `uv`.

As we've already built the region graph of the closure at this point and separately check that it is consistent, we are also able to assume the outlive constraints `uv: lower_bound` here.

So if we have a type-outlives bounds we can't prove, e.g. `T: 'local_infer`, we use the region graph to go to universal variables `'a` with `'a: local_infer`. In case `'a` are local, we then use the assumed outlived constraints to go to non-local ones.

We then store the list of promoted type tests in the `BorrowCheckResults`.
We then apply them in while borrow-checking its parent in `TypeChecker::prove_closure_bounds`.

TODO: explain how exactly that works :3
