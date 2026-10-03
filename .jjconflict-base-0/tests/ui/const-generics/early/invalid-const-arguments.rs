#![crate_type = "lib"]

struct A<const N: u8>;
trait Foo {}
impl Foo for A<N> {}
//~^ ERROR cannot find const `N` in this scope
//~| ERROR unresolved item provided when a constant was expected

struct B<const N: u8>;
impl<N> Foo for B<N> {}
//~^ ERROR type provided when a constant

struct C<const C: u8, const N: u8>;
impl<const N: u8> Foo for C<N, T> {}
//~^ ERROR cannot find const `T` in this scope
//~| ERROR unresolved item provided when a constant was expected

struct D<const E: u8, const X: u8, const P: u32>;
impl Foo for D<E, X, P> {}
//~^ ERROR cannot find const `E` in this scope
//~| ERROR unresolved item provided when a constant was expected
//~| ERROR cannot find const `X` in this scope
//~| ERROR cannot find const `P` in this scope
struct R<const O: u8, const G: u8, const F: u32>;
impl<const F: u8, const H: u32> Foo for D<F, Q, D> {}
//~^ ERROR cannot find const `Q` in this scope
//~| ERROR unresolved item provided when a constant was expected
