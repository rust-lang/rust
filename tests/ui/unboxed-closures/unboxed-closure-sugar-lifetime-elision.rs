// Test that the unboxed closure sugar can be used with an arbitrary
// struct type and that it is equivalent to the same syntax using
// angle brackets. This test covers only simple types and in
// particular doesn't test bound regions.

#![feature(unboxed_closures)]
#![allow(dead_code)]

use std::marker;

trait Foo<T> {
    type Output;
    fn dummy(&self, t: T);
}

trait Eq<X: ?Sized> { }
impl<X: ?Sized> Eq<X> for X { }
fn eq<A: ?Sized,B: ?Sized +Eq<A>>() { }

fn main() {
    eq::< dyn for<'a> Foo<(&'a isize,), Output=&'a isize>,
          dyn Foo(&isize) -> &isize                                   >();
    eq::< dyn for<'a> Foo<(&'a isize,), Output=(&'a isize, &'a isize)>,
          dyn Foo(&isize) -> (&isize, &isize)                           >();

    let _: dyn Foo(&isize, &usize) -> &usize; //~ ERROR missing lifetime specifier
}
