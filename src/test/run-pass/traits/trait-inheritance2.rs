// run-pass
#![allow(dead_code)]

trait Foo { fn f(&self) -> isize; }
trait Bar { fn g(&self) -> isize; }
trait Baz { fn h(&self) -> isize; }

trait Quux: Foo + Bar + Baz { }

struct A { x: isize }

impl Foo for A { fn f(&self) -> isize { 10 } }
impl Bar for A { fn g(&self) -> isize { 20 } }
impl Baz for A { fn h(&self) -> isize { 30 } }
impl Quux for A {}

fn f<T:Quux + Foo + Bar + Baz>(a: &T) {
    assert_eq!(a.f(), 10);
    assert_eq!(a.g(), 20);
    assert_eq!(a.h(), 30);
}

pub fn main() {
    let a = &A { x: 3 };
    f(a);
}
