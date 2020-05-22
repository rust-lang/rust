#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

trait Baz {
    type Quaks;
}
impl Baz for u8 {
    type Quaks = [u16; 3];
}

trait Bar {}
impl Bar for [u16; 4] {}
impl Bar for [[u16; 3]; 3] {}

trait Foo  //~ ERROR the trait bound `[u16; 3]: Bar` is not satisfied [E0277]
           //~^ ERROR the trait bound `[[u16; 3]; 2]: Bar` is not satisfied [E0277]
    where
        [<u8 as Baz>::Quaks; 2]: Bar,
        <u8 as Baz>::Quaks: Bar,
{
}

struct FooImpl;

impl Foo for FooImpl {}
//~^ ERROR the trait bound `[u16; 3]: Bar` is not satisfied [E0277]
//~^^ ERROR the trait bound `[[u16; 3]; 2]: Bar` is not satisfied [E0277]

fn f(_: impl Foo) {}
//~^ ERROR the trait bound `[u16; 3]: Bar` is not satisfied [E0277]
//~^^ ERROR the trait bound `[[u16; 3]; 2]: Bar` is not satisfied [E0277]

fn main() {
    f(FooImpl)
}
