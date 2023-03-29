#![feature(non_lifetime_binders)]
#![allow(incomplete_features)]

pub trait Trait {}

pub struct Wrapper<T: ?Sized>(Box<T>);

// @has non_lifetime_binders/fn.foo.html '//pre' "fn foo()where for<'a, T> &'a Wrapper<T>: Trait"
pub fn foo() where for<'a, T> &'a Wrapper<T>: Trait {}
