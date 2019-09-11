#![deny(warnings)]

pub struct Foo;

extern {
    pub fn foo(x: (Foo)); //~ ERROR `extern` block uses type `Foo`
}

fn main() {
}
