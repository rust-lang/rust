// check-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash
#![feature(lazy_normalization_consts)]
//~^ WARN the feature `lazy_normalization_consts` is incomplete and may cause the compiler to crash

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
