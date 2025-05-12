// Check that the `'_` used in structs/enums gives an error.

use std::fmt::Debug;

struct Foo {
    x: &'_ u32, //~ ERROR missing lifetime specifier
}

enum Bar {
    Variant(&'_ u32), //~ ERROR missing lifetime specifier
}

fn main() { }
