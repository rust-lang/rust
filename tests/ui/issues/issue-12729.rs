//@ edition: 2015
//@ check-pass
#![allow(dead_code)]

pub struct Foo;

mod bar {
    use Foo;

    impl Foo {
        fn baz(&self) {}
    }
}
fn main() {}
