// check-pass
// Regression test of #70944, should compile fine.

use std::ops::Index;

pub struct KeyA;
pub struct KeyB;
pub struct KeyC;

pub trait Foo: Index<KeyA> + Index<KeyB> + Index<KeyC> {}
pub trait FooBuilder {
    type Inner: Foo;
    fn inner(&self) -> &Self::Inner;
}

pub fn do_stuff(foo: &impl FooBuilder) {
    let inner = foo.inner();
    &inner[KeyA];
    &inner[KeyB];
    &inner[KeyC];
}

fn main() {}
