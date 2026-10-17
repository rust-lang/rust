//@ compile-flags: -Zassumptions-on-binders -Znext-solver=globally

#![feature(test_binder_constraints)]

trait Project {
    type Assoc;
}

impl<T> Project for T {
    type Assoc = T;
}

core::test_binder_constraints! {
    impl<T: Project<Assoc: 'static>> {
        forall<'a> {
            or {
                ambiguity,
                for<> T::Assoc: 'a,
            }
        }
    }
}

// If the concrete candidate is rejected at the root, the ambiguous alternative must
// still cause an error rather than disappearing with the rejected candidate.
core::test_binder_constraints! {
    impl<T: Project> {
        forall<'a> {
            //~^ ERROR unable to satisfy constraints involving placeholders
            or {
                ambiguity,
                for<> T::Assoc: 'a,
            }
        }
    }
}

// The order of the alternatives must not change whether the bound can be proved.
core::test_binder_constraints! {
    impl<T: Project<Assoc: 'static>> {
        forall<'a> {
            or {
                for<> T::Assoc: 'a,
                ambiguity,
            }
        }
    }
}

// A required ambiguous constraint still makes the whole AND ambiguous, even when
// its sibling is known to hold.
core::test_binder_constraints! {
    impl<T: Project<Assoc: 'static>> {
        forall<'a> {
            //~^ ERROR unable to satisfy constraints involving placeholders
            ambiguity,
            for<> T::Assoc: 'a,
        }
    }
}

// Ambiguity shared by every OR alternative is still required after canonicalization.
core::test_binder_constraints! {
    impl {
        forall<'a> {
            //~^ ERROR unable to satisfy constraints involving placeholders
            or {
                ambiguity,
                ambiguity,
            }
        }
    }
}

fn main() {}
