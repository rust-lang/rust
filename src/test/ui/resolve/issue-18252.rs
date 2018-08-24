enum Foo {
    Variant { x: usize }
}

fn main() {
    let f = Foo::Variant(42);
    //~^ ERROR expected function, found struct variant `Foo::Variant`
}
