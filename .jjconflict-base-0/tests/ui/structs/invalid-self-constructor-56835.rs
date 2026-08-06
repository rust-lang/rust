// https://github.com/rust-lang/rust/issues/56835
pub struct Foo {}

impl Foo {
    fn bar(Self(foo): Self) {}
    //~^ ERROR the `Self` constructor can only be used with tuple or unit structs
    //~^^ ERROR expected tuple struct or tuple variant, found self constructor `Self` [E0164]
}

fn main() {}
