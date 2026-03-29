#![feature(sized_hierarchy)]

use std::marker::SizeOfVal;

pub struct Foo<T> {
    inner: T,
}

pub trait Trait {
    fn foo(_: impl Sized);
    fn bar<T>(_: impl Sized)
    where
        Foo<T>: SizeOfVal;
    fn baz<'a, const N: usize>();
    fn quux<'a: 'b, 'b, T: ?Sized>();
}
