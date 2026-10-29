//! Regression test for <https://github.com/rust-lang/rust/issues/42312>.
//! Test unsized fn params are rejected.
//! (trait object, and `?Sized` assoc type)

use std::ops::Deref;

pub trait Foo {
    fn baz(_: Self::Target) where Self: Deref {}
    //~^ ERROR the size for values of type
}

pub fn f(_: dyn ToString) {}
//~^ ERROR the size for values of type

fn main() { }
