struct X<const N: u8>();

impl X<N> {}
//~^ ERROR cannot find type `N` in this scope
//~| ERROR unresolved item provided when a constant was expected
impl<T, const A: u8 = 2> X<N> {}
//~^ ERROR cannot find type `N` in this scope
//~| ERROR defaults for const parameters are only allowed in `struct`, `enum`, `type`, or `trait` definitions
//~| ERROR unresolved item provided when a constant was expected

fn foo(_: T) where T: Send {}
//~^ ERROR cannot find type `T` in this scope
//~| ERROR cannot find type `T` in this scope

fn bar<const N: u8>(_: A) {}
//~^ ERROR cannot find type `A` in this scope

fn main() {
}
