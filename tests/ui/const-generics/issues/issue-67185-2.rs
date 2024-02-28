trait Baz {
    type Quaks;
}
impl Baz for u8 {
    type Quaks = [u16; 3];
}

trait Bar {}
impl Bar for [u16; 4] {}
impl Bar for [[u16; 3]; 3] {}

trait Foo
where
    [<u8 as Baz>::Quaks; 2]: Bar, //~ ERROR trait `Bar` is not implemented for `[[u16; 3]; 2]`
    <u8 as Baz>::Quaks: Bar,  //~ ERROR trait `Bar` is not implemented for `[u16; 3]`
{
}

struct FooImpl;

impl Foo for FooImpl {}
//~^ ERROR trait `Bar` is not implemented for `[u16; 3]`
//~^^ ERROR trait `Bar` is not implemented for `[[u16; 3]; 2]`

fn f(_: impl Foo) {}
//~^ ERROR trait `Bar` is not implemented for `[u16; 3]`
//~^^ ERROR trait `Bar` is not implemented for `[[u16; 3]; 2]`

fn main() {
    f(FooImpl)
}
