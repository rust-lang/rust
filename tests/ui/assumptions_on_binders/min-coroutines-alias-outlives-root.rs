//@ check-pass
//@ compile-flags: -Zassumptions-on-binders=min_coroutines -Znext-solver=globally

#![feature(test_binder_constraints)]
#![allow(internal_features)]

trait Trait {
    type Assoc;
}

trait Bounded {
    type Assoc: 'static;
}

// Each of stable's alias-outlives rules must remain available at the root. In particular,
// an environment or item bound must not also require the alias's components to outlive.
core::test_binder_constraints! {
    impl<'a, T: Trait> where T::Assoc: 'a {
        predicates T::Assoc: 'a
    }
}

core::test_binder_constraints! {
    impl<T: Bounded> {
        predicates T::Assoc: 'static
    }
}

core::test_binder_constraints! {
    impl<'a, T: Trait + 'a> {
        predicates T::Assoc: 'a
    }
}

// Checking the signature also needs the implied outlives bound on the alias.
fn implied<'a, T: Trait>(_: &'a T::Assoc) {}

trait Higher<'a> {
    type Assoc: 'static;
}
trait Check<'a, 'b> {}
impl<'a, 'b, T: Higher<'b>> Check<'a, 'b> for T where T::Assoc: 'a {}

trait Outer<'a> {}
impl<'a, T> Outer<'a> for T where T: for<'b> Check<'a, 'b> {}

// These ordinary higher-ranked binders are entered by the solver, so they have no coroutine
// assumptions. The alias must survive both universes and use its item bound at the root.
core::test_binder_constraints! {
    impl<T: for<'b> Higher<'b>> {
        predicates T: for<'a> Outer<'a>
    }
}

fn main() {}
