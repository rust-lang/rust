//@ check-pass
#![feature(pin_ergonomics)]
#![allow(dead_code, incomplete_features)]

// Handle the case where there's ambiguity between pin as a contextual keyword and pin as a path.

struct Foo;

mod pin {
    pub struct Foo;
}

fn main() {
    let _x: &pin ::Foo = &pin::Foo;
}
