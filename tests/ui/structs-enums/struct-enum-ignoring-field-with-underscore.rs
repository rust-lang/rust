enum Foo {
    Bar { bar: bool },
    Other,
}

fn main() {
    let foo = Some(Foo::Other);

    if let Some(Foo::Bar {_}) = foo {}
    //~^ ERROR expected identifier, found reserved identifier `_`
    //~| ERROR pattern does not mention field `bar` [E0027]
}
