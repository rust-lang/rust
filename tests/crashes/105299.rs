//@ known-bug: #105299

pub trait Foo: Clone {}

pub struct Bar<'a, T: Clone> {
    pub cow: std::borrow::Cow<'a, [T]>,

    pub THIS_CAUSES_ICE: (), // #1
}

impl<T> Bar<'_, T>
where
    T: Clone,
    [T]: Foo,
{
    pub fn MOVES_SELF(self) {} // #2
}

pub fn main() {}
