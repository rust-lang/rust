// Regression test for https://github.com/rust-lang/rust/issues/148630
//
// We previously ran into ICE in wfcheck diagnostics code.
// After replacing unconstained infers and non-rigid aliases with `Ty/Const/Region::Error`
// in param env normalization, this no longer ICEs.

#![feature(unboxed_closures)]

trait Tr {}
trait Foo {
    fn foo() -> impl Sized
    //~^ ERROR: expected an `FnOnce<&'a i32>` closure, found `()`
    //~| ERROR: expected an `FnOnce<&'a i32>` closure, found `()`
    where
        for<'a> <() as FnOnce<&'a i32>>::Output: Tr,
    {
    }
}

fn main() {}
