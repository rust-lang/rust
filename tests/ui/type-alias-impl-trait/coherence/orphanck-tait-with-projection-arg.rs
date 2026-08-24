// regression test for #148621
type Foo<V> = impl std::fmt::Debug; //~ ERROR `impl Trait` in type aliases is unstable
//~| ERROR unconstrained opaque type

trait Identity<Q> {
    type T;
}

impl<Q> Clone for Foo<<() as Identity<Q>>::T> { //~ ERROR type parameter `Q` must be used as an argument to some local type
//~| ERROR the type parameter `Q` is not constrained by the impl trait, self type, or predicates
    fn clone(&self) -> Self {
        //~^ ERROR the trait bound `(): Identity<Q>` is not satisfied
        loop {}
    }
}

fn main() {}
