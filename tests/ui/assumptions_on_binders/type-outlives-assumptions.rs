//@ check-pass
//@ compile-flags: -Zassumptions-on-binders

#![feature(test_binder_constraints, non_lifetime_binders)]
#![expect(incomplete_features)]

// A type outlives assumption also implies that each region in the type outlives the RHS. The
// unrelated `'c` makes the derived edge visible in the lifted result. Its bounds then satisfy the
// lifted constraint after the outer binder is left.
core::test_binder_constraints! {
    impl<'b, 'c: 'b + 'static> {
        forall<'a> where &'b u8: 'a {
            'c: 'a
        } expect {
            or {
                'c: 'b,
                'c: 'static,
            }
        }
    }
}

// Regions bound inside the type are ignored, but free regions still contribute outlives edges.
core::test_binder_constraints! {
    impl<'b, 'd: 'b + 'static> {
        forall<'a> where for<'c> fn(&'c (), &'b u8): 'a {
            'd: 'a
        } expect {
            or {
                'd: 'b,
                'd: 'static,
            }
        }
    }
}

fn main() {}
