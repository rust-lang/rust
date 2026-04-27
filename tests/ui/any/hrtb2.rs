//@ revisions: next old
//@[next] compile-flags: -Znext-solver
#![feature(try_as_dyn)]

use std::any::try_as_dyn;

trait Foo<'a> {}

trait Bar {}
impl<T: ?Sized> Bar for Option<*const T> where T: for<'a> Foo<'a> {}

const _: () = {
    let x: Option<*const dyn Foo<'_>> = None;
    let _dy: &dyn Bar = try_as_dyn(&x).unwrap();
    //~^ ERROR: `Option::unwrap()` on a `None` value
};

fn main() {}
