// check-pass
// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

trait Baz {
    type Quaks;
}
impl Baz for u8 {
    type Quaks = [u16; 3];
}

trait Bar {}
impl Bar for [u16; 3] {}
impl Bar for [[u16; 3]; 2] {}

trait Foo
    where
        [<u8 as Baz>::Quaks; 2]: Bar,
        <u8 as Baz>::Quaks: Bar,
{
}

struct FooImpl;

impl Foo for FooImpl {}

fn f(_: impl Foo) {}

fn main() {
    f(FooImpl)
}
