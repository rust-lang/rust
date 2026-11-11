//@ check-pass
#![warn(redundant_imports)]

pub mod bar {
    pub struct Foo(pub Bar);
    pub struct Bar(pub char);
}

use bar::*;

pub fn warning() -> Foo {
    use bar::Foo; //~ WARNING imported redundantly
    Foo(Bar('a'))
}

fn main() {}
