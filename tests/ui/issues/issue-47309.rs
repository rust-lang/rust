// Make sure that the mono-item collector does not crash when trying to
// instantiate a default impl of a method with lifetime parameters.
// See https://github.com/rust-lang/rust/issues/47309

//@ compile-flags:-Clink-dead-code
//@ build-pass

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
