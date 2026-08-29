//@ compile-flags: -Zassumptions-on-binders
//@ check-pass

// Regression test for #161733.
// Previously we didn't properly handle identity region outlives
// and it was mistakenly considered as false region constraint.
// The closure is needed to register the type outlives obligation
// in borrowck.

#![feature(test_binder_constraints)]

fn foo<T>()
where
    for<'a> &'a T: 'a,
{
    || {};
}

core::test_binder_constraints! {
    impl<T> {
        forall<'a> where T: 'a {
            'a: 'a,
            T: 'a,
        } expect {
        }
    }
}

fn main() {}
