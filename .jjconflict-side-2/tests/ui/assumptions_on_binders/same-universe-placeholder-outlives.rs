//@ check-pass
//@ compile-flags: -Zassumptions-on-binders

#![feature(test_binder_constraints, non_lifetime_binders)]
#![expect(incomplete_features)]

// Regression test for rust-lang/project-assumptions-on-binders#19.
//
// When leaving a binder we lift its region constraints into a smaller universe. Constraints
// which already hold inside of the binder have no lower universe candidates to be lifted to,
// so they used to turn into `Or([])`, i.e. `false`. They have to be discharged instead.

// Outlives is reflexive.
core::test_binder_constraints! {
    impl<> {
        forall<'a> {
            'a: 'a
        } expect {
        }
    }
}

// Directly entailed by an assumption of the binder we're leaving.
core::test_binder_constraints! {
    impl<> {
        forall<'a, 'b> where 'b: 'a {
            'b: 'a
        } expect {
        }
    }
}

// Transitively entailed by the assumptions of the binder we're leaving.
core::test_binder_constraints! {
    impl<> {
        forall<'a, 'b, 'c> where 'c: 'b, 'b: 'a {
            'c: 'a
        } expect {
        }
    }
}

// Discharging entailed constraints must not swallow the ones which still have to be lifted
// into the outer universe. Here `'a: 'a` and `'b: 'a` are discharged inside the binder while
// `'c: 'a` is lifted, as `'c` outlives every lower universe region that `'a` outlives.
//
// FIXME(-Zassumptions-on-binders): this should be `impl<'b, 'c: 'b>`, not
// `impl<'b, 'c: 'b + 'static>`, but OR isn't actually implemented yet
core::test_binder_constraints! {
    impl<'b, 'c: 'b + 'static> {
        forall<'a> where 'b: 'a {
            'a: 'a,
            'b: 'a,
            'c: 'a,
        } expect {
            or {
                'c: 'b,
                'c: 'static,
            }
        }
    }
}

fn main() {}
