pub struct Foo;

pub trait Bar {
    #[doc(hidden)]
    fn bar() {}
}

impl Bar for Foo {
    fn bar() {}
}
