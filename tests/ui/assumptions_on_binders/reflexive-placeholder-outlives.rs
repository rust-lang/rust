//@ check-pass
//@ compile-flags: -Zassumptions-on-binders

#![feature(test_binder_constraints, non_lifetime_binders)]
#![expect(incomplete_features)]

// Regression test for rust-lang/project-assumptions-on-binders#19.
core::test_binder_constraints! {
    impl<> {
        forall<'a> {
            'a: 'a
        } expect {
        }
    }
}

fn main() {}
