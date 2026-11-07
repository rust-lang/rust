#![feature(sized_hierarchy)]

use std::marker::MetaSized;

pub struct Foo<T> {
    inner: T,
}

pub trait Trait {
    fn foo(_: impl Sized);
    fn bar<T>(_: impl Sized)
    where
        Foo<T>: MetaSized;
    fn baz<'a, const N: usize>();
    fn quux<'a: 'b, 'b, T: ?Sized>();
}
