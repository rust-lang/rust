//@ revisions: pass fail
//@[pass] check-pass
//@ compile-flags: -Zassumptions-on-binders

#![feature(test_binder_constraints, non_lifetime_binders)]
#![expect(incomplete_features)]

trait Project {
    type Assoc;
}

impl<T> Project for T {
    type Assoc = T;
}

// The alias involving U cannot leave its non-lifetime binder and becomes ambiguous.
// Keep the other alias candidate until the root, where its outlives assumption is known.
#[cfg(pass)]
core::test_binder_constraints! {
    impl<T: Project<Assoc: 'static>> {
        forall<'a, U> {
            or {
                for<> <U as Project>::Assoc: 'a,
                for<> T::Assoc: 'a,
            }
        }
    }
}

// If the concrete candidate is rejected at the root, the ambiguous alternative must
// still cause an error rather than disappearing with the rejected candidate.
#[cfg(fail)]
core::test_binder_constraints! {
    impl<T: Project> {
        forall<'a, U> {
            //[fail]~^ ERROR unable to satisfy constraints involving placeholders
            or {
                for<> <U as Project>::Assoc: 'a,
                for<> T::Assoc: 'a,
            }
        }
    }
}

// The order of the alternatives must not change whether the bound can be proved.
#[cfg(pass)]
core::test_binder_constraints! {
    impl<T: Project<Assoc: 'static>> {
        forall<'a, U> {
            or {
                for<> T::Assoc: 'a,
                for<> <U as Project>::Assoc: 'a,
            }
        }
    }
}

// A required ambiguous constraint still makes the whole AND ambiguous, even when
// its sibling is known to hold.
#[cfg(fail)]
core::test_binder_constraints! {
    impl<T: Project<Assoc: 'static>> {
        forall<'a, U> {
            //[fail]~^ ERROR unable to satisfy constraints involving placeholders
            for<> <U as Project>::Assoc: 'a,
            for<> T::Assoc: 'a,
        }
    }
}

fn main() {}
