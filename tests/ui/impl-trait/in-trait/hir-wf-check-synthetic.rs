//! Regression test for <https://github.com/rust-lang/rust/issues/148630>.
//! An RPITIT method with a where-clause that references a trait via an
//! associated type projection must not ICE in HIR wf-checking.

#![feature(unboxed_closures)]

trait Tr {}
trait Foo {
    fn foo() -> impl Sized
    //~^ ERROR E0277
    //~| ERROR E0277
    //~| ERROR E0277
    //~| ERROR E0277
    //~| ERROR E0277
    //~| ERROR E0277
    //~| ERROR E0277
    where
        for<'a> <() as FnOnce<&'a i32>>::Output: Tr,
    {
    }
}

fn main() {}
