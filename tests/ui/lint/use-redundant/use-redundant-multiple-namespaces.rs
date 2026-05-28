//@ check-pass
#![allow(nonstandard_style)]

pub mod bar {
    pub struct Foo { pub bar: Bar }
    pub struct Bar(pub char);
}

pub mod x {
    use crate::bar;
    pub const Foo: bar::Bar = bar::Bar('a');
}

pub fn warning() -> bar::Foo {
    #![deny(unused_imports)] // no error
    use bar::*;
    use x::Foo;
    Foo { bar: Foo }
}

fn main() {}
