use std::fmt::Debug;

trait Foo {
    fn baz(&self) -> impl Debug;
}

impl Foo for u32 {
    fn baz(&self) -> impl Debug {
        32
    }
}

fn main() {
    let i = Box::new(42_u32) as Box<dyn Foo>;
    //~^ ERROR the trait `Foo` is not dyn compatible
    let s = i.baz();
}
