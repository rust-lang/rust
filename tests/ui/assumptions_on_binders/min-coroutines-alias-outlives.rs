//@ compile-flags: -Zassumptions-on-binders=min_coroutines -Znext-solver=globally
//@ normalize-stderr: "\[[0-9a-f]{4}\]" -> "[HASH]"

#![feature(test_binder_constraints)]
#![allow(internal_features)]

trait Trait {
    type Assoc;
}

// Ordinary types are eagerly decomposed, so the assumption on T discharges the tuple's
// component. This temporarily strengthens the eager leak check until VisibleForLeakCheck
// can distinguish the resulting constraints.
core::test_binder_constraints! {
    impl<T> {
        forall<'a> where T: 'a {
            predicates (T,): 'a
        } expect {}
    }
}

// Aliases stay intact even with an assumption naming the same alias. The actual output must
// contain AliasOutlives, not an OR of item bounds, environment bounds, and components.
core::test_binder_constraints! {
    impl<T: Trait> {
        forall<'a> where T::Assoc: 'a {
            //~^ ERROR forall expect clause failed
            //~| ERROR may not live long enough
            predicates T::Assoc: 'a
        } expect {}
    }
}

// Component bounds must not discharge aliases either. Retain the inner placeholder while
// leaving both binders, even when the outer binder has no relevant assumptions.
core::test_binder_constraints! {
    impl<T: Trait> {
        forall<'outer> where T: 'outer {
            //~^ ERROR forall expect clause failed
            //~| ERROR may not live long enough
            forall<'inner> where T: 'inner {
                //~^ ERROR forall expect clause failed
                predicates T::Assoc: 'inner
            } expect {}
        } expect {}
    }
}

fn main() {}
