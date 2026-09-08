//@ compile-flags: -Zassumptions-on-binders=min_coroutines
//@ normalize-stderr: "\[[0-9a-f]{4}\]" -> "[HASH]"

#![feature(test_binder_constraints)]
#![allow(internal_features)]

trait Trait {
    type Assoc;
}

// Minimal mode keeps type outlives constraints intact instead of destructuring them into their
// components, so these constraints are retained and left for the root inference context.
//
// The `actual` constraint in the expected output is the point of these tests, so do not normalize
// it away: it is what distinguishes the retained `TypeOutlives` leaf from the OR of item bounds,
// env assumptions and components that eager destructuring would produce.

// The assumption names the component `T` while the goal names the composite `(T,)`. Destructuring
// eagerly would reduce the goal to its component and discharge it against the assumption, which is
// exactly the strengthening of the eager leak check that this mode avoids. Keeping the constraint
// whole means it is retained instead, so this `expect` clause fails.
core::test_binder_constraints! {
    impl<T> {
        forall<'a> where T: 'a {
            //~^ ERROR forall expect clause failed
            where (T,): 'a
        } expect {}
    }
}

// FIXME(-Zassumptions-on-binders): the assumption on the binder names the very same alias, so this
// ought to be discharged and the `expect` clause ought to hold. It is not, because the assumption
// is lowered without being normalized and so carries a non-rigid alias, while the goal is
// normalized to a rigid one, and the two do not compare equal. See the FIXME about normalizing
// assumptions in `region_assumptions_for_placeholders_in_universe`. Destructuring the constraint
// eagerly would lose the `TypeOutlives` leaf that this matching needs, which is why minimal mode
// keeps it.
core::test_binder_constraints! {
    impl<T: Trait> {
        forall<'a> where T::Assoc: 'a {
            //~^ ERROR forall expect clause failed
            where T::Assoc: 'a
        } expect {}
    }
}

fn main() {}
