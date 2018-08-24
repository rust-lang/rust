#![feature(unboxed_closures)]

struct Foo;

impl Fn<(&(),)> for Foo { } //~ ERROR missing lifetime specifier

fn main() {}
