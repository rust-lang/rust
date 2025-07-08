use std::num::NonZero;

trait Foo {}

impl Foo for u8 {}
impl Foo for u16 {}

fn foo(_: impl Foo) {}

fn main() {
    let x = NonZero::new(5_u8).unwrap();
    foo(x as _);
    //~^ ERROR: type annotations needed
}
