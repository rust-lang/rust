//@ check-pass

#![feature(return_type_notation)]
//~^ WARN the feature `return_type_notation` is incomplete

trait Foo {
    fn method() -> impl Sized;
}

trait Bar: Foo {
    fn other()
    where
        Self::method(..): Send;
}

fn is_send(_: impl Send) {}

impl<T: Foo> Bar for T {
    fn other()
    where
        Self::method(..): Send,
    {
        is_send(Self::method());
    }
}

fn main() {}
