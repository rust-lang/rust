enum Foo {
    Variant { x: usize }
}

fn main() {
    let f = Foo::Variant(42);
    //~^ ERROR expected value, found struct variant `Foo::Variant`
}
