// Regression test for #148630 (simpler reproduction).

#![feature(unboxed_closures)]

trait Tr {}
trait Foo {
    fn foo() -> impl Sized
    //~^ ERROR expected a `FnOnce<&'a i32>` closure, found `()`
    //~| ERROR expected a `FnOnce<&'a i32>` closure, found `()`
    //~| ERROR expected a `FnOnce<&'a i32>` closure, found `()`
    //~| ERROR expected a `FnOnce<&'a i32>` closure, found `()`
    //~| ERROR expected a `FnOnce<&'a i32>` closure, found `()`
    //~| ERROR expected a `FnOnce<&'a i32>` closure, found `()`
    //~| ERROR expected a `FnOnce<&'a i32>` closure, found `()`
    where
        for<'a> <() as FnOnce<&'a i32>>::Output: Tr,
    {
    }
}

fn main() {}
