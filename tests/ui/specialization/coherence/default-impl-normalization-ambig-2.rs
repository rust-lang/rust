// regression test for #118987
#![feature(specialization)] //~ WARN the feature `specialization` is incomplete

trait Assoc {
    type Output;
}

default impl<T: Clone> Assoc for T {
    type Output = bool;
}

impl Assoc for u8 {}

trait Foo {}

impl Foo for <u8 as Assoc>::Output {}
impl Foo for <u16 as Assoc>::Output {}
//~^ ERROR the trait bound `u16: Assoc` is not satisfied
fn main() {}
