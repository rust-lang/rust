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

fn main() {}
