//@ check-pass
// https://github.com/rust-lang/rust/issues/56593#issue-388659456

struct Foo;

mod foo {
    use super::*;

    #[derive(Debug)]
    pub struct Foo;
}

mod bar {
    use super::foo::*;

    fn bar(_: Foo) {}
}

fn main() {}
