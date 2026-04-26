//@ run-pass
#![allow(warnings)]

#[derive(Debug)]
pub struct Foo<T>(pub T);

use std::fmt;

impl<T> Field for T {}
impl<T> Finish for T {}
impl Dt for &mut fmt::Formatter<'_> {}

pub trait Field {
    fn field(&self, _: impl Sized) {
        panic!("got into field");
    }
}
pub trait Finish {
    fn finish(&self) -> Result<(), std::fmt::Error> {
        panic!("got into finish");
    }
}
pub trait Dt {
    fn debug_tuple(&self, _: &str) {
        panic!("got into debug_tuple");
    }
}

fn main() {
    let foo = Foo(());
    assert_eq!("Foo(())", format!("{:?}", foo));
}
