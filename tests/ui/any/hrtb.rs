//@ revisions: next old
//@[next] compile-flags: -Znext-solver
#![feature(try_as_dyn)]

use std::any::try_as_dyn;

trait Foo<'a, 'b> {}

trait Bar {}
impl<T: for<'a, 'b> Foo<'a, 'b> + ?Sized> Bar for Option<*const T> {}

const _: () = {
    let x: Option<*const dyn for<'a> Foo<'a, 'a>> = None;
    let _dy: &dyn Bar = try_as_dyn(&x).unwrap();
    //~^ ERROR: `Option::unwrap()` on a `None` value
};

fn main() {}
