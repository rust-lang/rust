#![deny(rustdoc::broken_intra_doc_links)]

// Test intra-doc links on trait implementations with generics
// regression test for issue #92662

use std::marker::PhantomData;

pub trait Bar<T> {
    fn bar(&self);
}

pub struct Foo<U>(PhantomData<U>);

impl<T, U> Bar<T> for Foo<U> {
    fn bar(&self) {}
}

//@ has generic_trait_impl/fn.main.html '//a[@href="struct.Foo.html#method.bar"]' 'Foo::bar'
/// link to [`Foo::bar`]
pub fn main() {}
