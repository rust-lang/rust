#![deny(warnings)]

pub struct Foo;

extern "C" {
    pub fn foo(x: (Foo)); //~ ERROR `extern` block uses type `Foo`
}

fn main() {}
