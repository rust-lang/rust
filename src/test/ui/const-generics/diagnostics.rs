#![crate_type="lib"]
#![feature(min_const_generics)]
#![allow(incomplete_features)]

struct A<const N: u8>;
trait Foo {}
impl Foo for A<N> {}
//~^ ERROR cannot find type
//~| unresolved item provided when a constant

struct B<const N: u8>;
impl<N> Foo for B<N> {}
//~^ ERROR type provided when a constant

struct C<const C: u8, const N: u8>;
impl<const N: u8> Foo for C<N, T> {}
//~^ ERROR cannot find type
//~| unresolved item provided when a constant
