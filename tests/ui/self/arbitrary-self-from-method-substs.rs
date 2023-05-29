#![feature(arbitrary_self_types)]

use std::ops::Deref;

struct Foo(u32);
impl Foo {
    fn get<R: Deref<Target=Self>>(self: R) -> u32 {
        self.0
    }
}

fn main() {
    let mut foo = Foo(1);
    foo.get::<&Foo>();
    //~^ ERROR mismatched types
}
