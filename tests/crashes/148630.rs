//@ known-bug: #148630
#![feature(unboxed_closures)]

trait Tr {}
trait Foo {
    fn foo() -> impl Sized
    where
        for<'a> <() as FnOnce<&'a i32>>::Output: Tr,
    {
    }
}

fn main() {}
