// check-pass
// compile-flags: -Wunused

// ensure there are no special warnings about uninhabited types
// when deriving Debug on an empty enum

#[derive(Debug)]
enum Void {}

#[derive(Debug)]
enum Foo {
    Bar(#[allow(dead_code)] u8),
    Void(#[allow(dead_code)] Void), //~ WARN never constructed
}

fn main() {
    let x = Foo::Bar(42);
    println!("{:?}", x);
}
