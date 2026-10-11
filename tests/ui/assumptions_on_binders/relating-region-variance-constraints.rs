//@ check-pass
//@ compile-flags: -Zassumptions-on-binders -Znext-solver=globally

#![feature(test_binder_constraints)]

core::test_binder_constraints! {
    impl<'a: 'b, 'b> {
        forall<> {
            'a: 'b
        } expect {
            'a: 'b
        }
    }
}

core::test_binder_constraints! {
    impl<'a, 'b: 'a> {
        forall<> {
            'b: 'a
        } expect {
            'b: 'a
        }
    }
}

core::test_binder_constraints! {
    impl<'a: 'b, 'b> {
        forall<'c> where 'c: 'b {
            forall<'d> where 'd: 'c {
                'd: 'b
            } expect {
                or {
                    'c: 'b,
                    'c: 'static,
                }
            }
        } expect {}
    }
}

fn main() {}
