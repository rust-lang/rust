use std::ops::DispatchFromDyn; //~ ERROR use of unstable library feature `dispatch_from_dyn`
struct Smaht<T, MISC>(PhantomData); //~ ERROR cannot find type `PhantomData` in this scope
impl<T> DispatchFromDyn<Smaht<U, MISC>> for T {} //~ ERROR cannot find type `U` in this scope
//~^ ERROR cannot find type `MISC` in this scope
//~| ERROR use of unstable library feature `dispatch_from_dyn`
//~| ERROR the trait `DispatchFromDyn` may only be implemented for a coercion between structures
trait Foo: X<u32> {}
trait X<T> {
    fn foo(self: Smaht<Self, T>); //~ ERROR: invalid `self` parameter type
}
trait Marker {}
impl Marker for dyn Foo {}
fn main() {}
