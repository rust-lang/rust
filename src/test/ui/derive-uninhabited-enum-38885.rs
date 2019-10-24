// build-pass (FIXME(62277): could be check-pass?)
// compile-flags: -Wunused

// ensure there are no special warnings about uninhabited types
// when deriving Debug on an empty enum

#[derive(Debug)]
enum Void {}

#[derive(Debug)]
enum Foo {
    Bar(u8),
    Void(Void), //~ WARN never constructed
}

fn main() {
    let x = Foo::Bar(42);
    println!("{:?}", x);
}
