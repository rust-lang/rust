// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
// pretty-expanded FIXME #23616

pub struct Foo;

mod bar {
    use Foo;

    impl Foo {
        fn baz(&self) {}
    }
}
fn main() {}
