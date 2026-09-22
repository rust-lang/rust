//@ check-fail
//@ compile-flags: -Zassumptions-on-binders

#![feature(test_binder_constraints, non_lifetime_binders)]
#![expect(incomplete_features)]

core::test_binder_constraints! {
    impl<'a: 'b, 'b> {
        'a: 'b
    }
}

core::test_binder_constraints! {
    impl<'a: 'b, 'b> {
        'a: 'b,
        forall { }
    }
}

// One satisfied alternative is enough to discharge the root constraint.
core::test_binder_constraints! {
    impl<'b, 'c: 'b> {
        forall<'a> where 'b: 'a {
            'c: 'a
        } expect {
            or {
                'c: 'b,
                'c: 'static,
            }
        }
    }
}

core::test_binder_constraints! {
    impl<'a, T: 'a> {
        T: 'a,
        forall<T2> where T2: 'a {
            T2: 'a,
        }
    }
}

trait Trait {
    type Assoc;
}

// The top-level constraint must be checked after leaving the binder.
// Regression for project-assumptions-on-binders#26.
//
// for<> syntax does direct insert into constraint storage
core::test_binder_constraints! {
    impl<T: Trait> {
        forall<'a> {
            //~^ ERROR unable to satisfy outlives constraints
            for<> T::Assoc: 'a
        } expect {
            or {
                for<'b> T::Assoc: 'b,
                for<> T::Assoc: 'static
            }
        }
    }
}

// The top-level constraint must be checked after leaving the binder.
// Regression for project-assumptions-on-binders#26.
//
// `where` syntax goes through the full clause destructuring and register_obligation pipeline
core::test_binder_constraints! {
    impl<T: Trait> {
        forall<'a> {
            //~^ ERROR unable to satisfy outlives constraints
            where T::Assoc: 'a
        } expect {
            or {
                for<'b> T::Assoc: 'b,
                for<> T::Assoc: 'static,
                T: 'static
            }
        }
    }
}

fn main() {}
