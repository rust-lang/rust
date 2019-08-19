// build-pass (FIXME(62277): could be check-pass?)
// compile-flags: -Wunused

// ensure there are no special warnings about uninhabited types
// when deriving Debug on an empty enum

#[derive(Debug)]
enum Void {} //~ WARN never used

#[derive(Debug)]
enum Foo { //~ WARN never used
    Bar(u8),
    Void(Void),
}

fn main() {}
