// Regression test for <https://github.com/rust-lang/rust/issues/144888>.
// This used to ICE with `unhandled node Crate(Mod)`.

#![crate_type = "lib"]

struct ActuallySuper;

trait Super<Q> {
    type Assoc;
}

trait Dyn {}
impl<T, U> Dyn for dyn Foo<T, U> + '_ {}
//~^ ERROR the trait `Foo` is not dyn compatible

trait Foo<T, U>: Super<ActuallySuper, Assoc = T>
where
    <Self as Mirror>::Assoc: Super,
    //~^ ERROR missing generics for trait `Super`
{
    fn transmute(&self, t: T) -> <Self as Super>::Assoc;
    //~^ ERROR missing generics for trait `Super`
}

trait Mirror {
    type Assoc: ?Sized;
}

impl<T: Super<ActuallySuper, Assoc = T>> Mirror for T {}
//~^ ERROR not all trait items implemented
