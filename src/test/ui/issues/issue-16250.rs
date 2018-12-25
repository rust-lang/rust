#![deny(warnings)]

pub struct Foo;

extern {
    pub fn foo(x: (Foo)); //~ ERROR unspecified layout
}

fn main() {
}
