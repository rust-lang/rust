// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
// pretty-expanded FIXME #23616

pub mod a {
    pub struct Foo { a: usize }
}

pub mod b {
    use a::Foo;
    impl Foo {
        fn bar(&self) { }
    }
}

pub fn main() { }
