#![warn(clippy::struct_excessive_bools)]

struct S {
    //~^ struct_excessive_bools
    a: bool,
}

struct Foo;

fn main() {}
