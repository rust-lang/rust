struct X<const N: u8>();

impl X<N> {}
//~^ ERROR cannot find type `N`
//~| ERROR unresolved item provided when a constant was expected
impl<T, const A: u8 = 2> X<N> {}
//~^ ERROR cannot find type `N`
//~| ERROR defaults for const parameters are only allowed in `struct`, `enum`, `type`, or `trait` definitions
//~| ERROR unresolved item provided when a constant was expected

fn foo(_: T) where T: Send {}
//~^ ERROR cannot find type `T`
//~| ERROR cannot find type `T`

fn bar<const N: u8>(_: A) {}
//~^ ERROR cannot find type `A`

fn main() {
}
