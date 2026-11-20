//@ run-pass
#![allow(dead_code)]

trait Foo { fn f(&self) -> isize; }
trait Bar : Foo { fn g(&self) -> isize; }

struct A { x: isize }

impl Foo for A { fn f(&self) -> isize { 10 } }

impl Bar for A {
    // Testing that this impl can call the impl of Foo
    fn g(&self) -> isize { self.f() }
}

pub fn main() {
    let a = &A { x: 3 };
    assert_eq!(a.g(), 10);
}
