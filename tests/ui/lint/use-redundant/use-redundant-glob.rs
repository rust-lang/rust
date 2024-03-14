//@ check-pass
#![warn(unused_imports)]

pub mod bar {
    pub struct Foo(pub Bar);
    pub struct Bar(pub char);
}

pub fn warning() -> bar::Foo {
    use bar::Foo; //~ WARNING redundant import: `bar::Foo`
    use bar::*;
    Foo(Bar('a'))
}

fn main() {}
