// Test that we don't ICE for a typeck error that only shows up in dropck
// Version where the normalization error is an ambiguous trait implementation.
// <[T] as ToOwned>::Owned is ambiguous on whether to use T: Clone or [T]::Clone.
// Regression test for #105299

pub trait Foo: Clone {}

pub struct Bar<'a, T: Clone> {
    pub cow: std::borrow::Cow<'a, [T]>,

    pub THIS_CAUSES_ICE: (),
}

impl<T> Bar<'_, T>
where
    T: Clone,
    [T]: Foo,
{
    pub fn MOVES_SELF(self) {}
    //~^ ERROR type annotations needed
}

pub fn main() {}
