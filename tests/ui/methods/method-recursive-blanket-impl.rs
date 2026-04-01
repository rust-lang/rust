//@ run-pass
#![allow(unused_variables)]
#![allow(unused_imports)]
// Test that we don't trigger on the blanket impl for all `&'a T` but
// rather keep autoderefing and trigger on the underlying impl.  To
// know not to stop at the blanket, we have to recursively evaluate
// the `T:Foo` bound.


use std::marker::Sized;

// Note: this must be generic for the problem to show up
trait Foo<A> { //~ WARN trait `Foo` is never used
    fn foo(&self, a: A);
}

impl Foo<u8> for [u8] {
    fn foo(&self, a: u8) {}
}

impl<'a, A, T> Foo<A> for &'a T where T: Foo<A> {
    fn foo(&self, a: A) {
        Foo::foo(*self, a)
    }
}

trait Bar {
    fn foo(&self);
}

struct MyType;

impl Bar for MyType {
    fn foo(&self) {}
}

fn main() {
    let mut m = MyType;
    (&mut m).foo()
}
