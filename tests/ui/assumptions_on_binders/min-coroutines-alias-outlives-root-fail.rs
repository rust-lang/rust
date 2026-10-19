//@ compile-flags: -Zassumptions-on-binders=min_coroutines -Znext-solver=globally

#![feature(test_binder_constraints)]
#![allow(internal_features)]

trait Trait {
    type Assoc;
}

// A deferred alias constraint still has to be checked at the root.
core::test_binder_constraints! {
    impl<'a, T: Trait> {
        predicates T::Assoc: 'a
        //~^ ERROR may not live long enough
    }
}

trait Higher<'a> {
    type Assoc;
}
trait Check<'a, 'b> {}
impl<'a, 'b, T: Higher<'b>> Check<'a, 'b> for T where T::Assoc: 'a {}

trait Outer<'a> {}
impl<'a, T> Outer<'a> for T where T: for<'b> Check<'a, 'b> {}

// Neither exiting an ordinary binder nor exiting its enclosing binder can drop the alias.
core::test_binder_constraints! {
    impl<T: for<'b> Higher<'b>> {
        predicates T: for<'a> Outer<'a>
        //~^ ERROR may not live long enough
    }
}

fn main() {}
