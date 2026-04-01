//@ check-pass

pub struct Foo(bar::Bar);

pub mod bar {
    pub struct Foo(pub Bar);
    pub struct Bar(pub char);
}

pub fn warning() -> Foo {
    use bar::*;
    #[deny(unused_imports)]
    use self::Foo; // no error
    Foo(Bar('a'))
}

fn main() {}
