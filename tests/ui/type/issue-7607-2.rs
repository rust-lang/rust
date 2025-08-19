//@ check-pass
#![allow(dead_code)]

pub mod a {
    pub struct Foo { a: usize }
}

pub mod b {
    use crate::a::Foo;
    impl Foo {
        fn bar(&self) { }
    }
}

pub fn main() { }
