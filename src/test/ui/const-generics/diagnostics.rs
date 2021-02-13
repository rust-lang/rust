#![crate_type="lib"]
#![feature(const_generics)]
#![allow(incomplete_features)]

struct A<const N: u8>;
trait Foo {}
impl Foo for A<N> {}
//~^ ERROR type provided when a constant
//~| ERROR cannot find type

struct B<const N: u8>;
impl<N> Foo for B<N> {}
//~^ ERROR type provided when a constant
