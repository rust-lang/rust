//@ check-pass
#![warn(unused_imports)]

pub mod bar {
    pub struct Foo(pub Bar);
    pub struct Bar(pub char);
}

use bar::*;

pub fn warning() -> Foo {
    use bar::Foo; //~ WARNING redundant import
    Foo(Bar('a'))
}

fn main() {}
