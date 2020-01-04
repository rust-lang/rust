#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete and may cause the compiler to crash

trait Baz {
    type Quaks;
}
impl Baz for u8 {
    type Quaks = [u16; 3];
}

trait Bar {}
impl Bar for [u16; 4] {}
impl Bar for [[u16; 3]; 3] {}

trait Foo  //~ ERROR mismatched types
    where
        [<u8 as Baz>::Quaks; 2]: Bar,
        <u8 as Baz>::Quaks: Bar,
{
}

struct FooImpl;

impl Foo for FooImpl {}
//~^ ERROR mismatched types
//~^^ ERROR mismatched types

fn f(_: impl Foo) {}
//~^ ERROR mismatched types
//~^^ ERROR mismatched types

fn main() {
    f(FooImpl)
}
