//@ check-pass
// https://github.com/rust-lang/rust/issues/34423

pub struct Foo;

pub trait Bar {
    #[doc(hidden)]
    fn bar() {}
}

impl Bar for Foo {
    fn bar() {}
}
