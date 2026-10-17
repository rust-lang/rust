//@ run-pass
#![allow(dead_code)]

trait Foo { fn f(&self) -> isize; }
trait Bar : Foo { fn g(&self) -> isize; }

struct A { x: isize }

impl Foo for A { fn f(&self) -> isize { 10 } }
impl Bar for A { fn g(&self) -> isize { 20 } }

fn ff<T:Foo>(a: &T) -> isize {
    a.f()
}

fn gg<T:Bar>(a: &T) -> isize {
    a.g()
}

pub fn main() {
    let a = &A { x: 3 };
    assert_eq!(ff(a), 10);
    assert_eq!(gg(a), 20);
}
