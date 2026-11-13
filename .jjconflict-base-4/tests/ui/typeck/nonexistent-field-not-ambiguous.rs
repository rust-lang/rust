struct Foo {
    val: MissingType,
    //~^ ERROR cannot find type `MissingType` in this scope
}

fn main() {
    Foo { val: Default::default() };
}
