#[repr(u8)]
enum Foo {
    X = 0,
    _ = 1, //~ ERROR unnamed enum variants are experimental
}

// This should not parse as an unnamed enum variant.
#[cfg(false)]
struct Foo {
    _: i32, //~ ERROR expected identifier, found reserved identifier `_`
}

fn main() {}
