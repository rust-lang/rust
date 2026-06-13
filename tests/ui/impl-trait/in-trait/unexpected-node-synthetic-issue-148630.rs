// Regression test for #148630.

#![feature(unboxed_closures)]

use std::future::Future;

trait Foo {
    fn foo() -> impl Sized
    //~^ ERROR expected a `FnOnce(&'a mut i32)` closure, found `(dyn Foo + 'static)`
    //~| ERROR expected a `FnOnce(&'a mut i32)` closure, found `(dyn Foo + 'static)`
    //~| ERROR expected a `FnOnce(&'a mut i32)` closure, found `(dyn Foo + 'static)`
    //~| ERROR expected a `FnOnce(&'a mut i32)` closure, found `(dyn Foo + 'static)`
    //~| ERROR expected a `FnOnce(&'a mut i32)` closure, found `(dyn Foo + 'static)`
    //~| ERROR expected a `FnOnce(&'a mut i32)` closure, found `(dyn Foo + 'static)`
    //~| ERROR expected a `FnOnce(&'a mut i32)` closure, found `(dyn Foo + 'static)`
    where
        for<'a> <dyn Foo as FnOnce<(&'a mut i32,)>>::Output: Future<Output = ()> + 'a,
        //~^ ERROR the trait `Foo` is not dyn compatible [E0038]
    {
    }
}

fn main() {}
