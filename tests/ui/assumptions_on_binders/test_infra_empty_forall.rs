//@ check-pass
//@ compile-flags: -Zassumptions-on-binders -Znext-solver=globally

#![feature(test_binder_constraints, non_lifetime_binders)]
#![expect(incomplete_features)]

core::test_binder_constraints! {
    impl<'a: 'b, 'b> {
        // This forall still gets eagerly handled by the testing DSL even though its bound vars
        // are unused. If we as a performance optimization don't create a new universe for this
        // forall then we'll wind up ICEing when trying to eagerly handle placeholders mistakenly
        // in the root universe
        forall<'c> {
            'a: 'b
        }
    }
}

fn main() {}
