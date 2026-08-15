//@ check-pass

pub mod bar {
    pub struct Foo(pub Bar);
    pub struct Bar(pub char);
}

pub mod x {
    pub struct Foo(pub crate::bar::Bar);
}

pub fn warning() -> x::Foo {
    use bar::*;
    #[deny(unused_imports)]
    use x::Foo; // no error
    Foo(Bar('a'))
}

fn main() {}
