// Make sure that the mono-item collector does not crash when trying to
// instantiate a default impl of a method with lifetime parameters.

// compile-flags:-Clink-dead-code
// compile-pass

#![crate_type="rlib"]

pub trait EnvFuture {
    type Item;

    fn boxed_result<'a>(self) where Self: Sized, Self::Item: 'a, {
    }
}

struct Foo;

impl<'a> EnvFuture for &'a Foo {
    type Item = ();
}
