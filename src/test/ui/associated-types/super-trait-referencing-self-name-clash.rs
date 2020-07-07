trait Foo {
    type Bar;
}
trait Qux: Foo + AsRef<Self::Bar> { //~ ERROR ambiguous associated type `Bar` in bounds of `Self`
    type Bar;
}

fn main() {}
