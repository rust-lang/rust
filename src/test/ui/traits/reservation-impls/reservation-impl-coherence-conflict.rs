// compile-fail

// check that reservation impls are accounted for in negative reasoning.

#![feature(rustc_attrs)]

trait MyTrait {}
#[rustc_reservation_impl]
impl MyTrait for () {}

trait OtherTrait {}
impl OtherTrait for () {}
impl<T: MyTrait> OtherTrait for T {}
//~^ ERROR conflicting implementations

fn main() {}
