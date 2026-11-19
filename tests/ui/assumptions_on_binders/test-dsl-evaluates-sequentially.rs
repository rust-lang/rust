//@ check-pass
//@ compile-flags: -Zassumptions-on-binders -Znext-solver=globally

#![feature(test_binder_constraints)]
core::test_binder_constraints! {
    impl<'a: 'b, 'b, 'c: 'd, 'd> {
        forall<'x> {
            'a: 'b,
            forall<'y> { } expect { 'a: 'b },
            'c: 'd,
        }
    }
}

fn main() {}
