struct Foo {
    val: MissingType,
    //~^ ERROR cannot find type `MissingType`
}

fn main() {
    Foo { val: Default::default() };
}
