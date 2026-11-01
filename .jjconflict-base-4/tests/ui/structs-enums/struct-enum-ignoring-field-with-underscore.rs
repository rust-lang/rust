enum Foo {
    Bar { bar: bool },
    Other,
}

fn main() {
    let foo = Some(Foo::Other);

    if let Some(Foo::Bar {_}) = foo {}
    //~^ ERROR expected field pattern, found `_`
}
