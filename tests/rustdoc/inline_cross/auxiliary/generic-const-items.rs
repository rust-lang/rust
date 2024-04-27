#![feature(generic_const_items)]
#![allow(incomplete_features)]

pub const K<'a, T: 'a + Copy, const N: usize>: Option<[T; N]> = None
where
    String: From<T>;

pub trait Trait<T: ?Sized> {
    const C<'a>: &'a T
    where
        T: 'a + Eq;
}

pub struct Implementor;

impl Trait<str> for Implementor {
    const C<'a>: &'a str = "C"
    // In real code we could've left off this bound but adding it explicitly allows us to test if
    // we render where-clauses on associated consts inside impl blocks correctly.
    where
        str: 'a;
}
