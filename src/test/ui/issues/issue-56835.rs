pub struct Foo {}

impl Foo {
    fn bar(Self(foo): Self) {}
    //~^ ERROR the `Self` constructor can only be used with tuple or unit structs
    //~^^ ERROR expected tuple struct/variant, found self constructor `Self` [E0164]
}

fn main() {}
