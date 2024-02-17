//@ run-pass
#![allow(dead_code)]

trait Foo { fn f(&self) -> isize; }
trait Bar : Foo { fn g(&self) -> isize; }
trait Baz : Bar { fn h(&self) -> isize; }

struct A { x: isize }

impl Foo for A { fn f(&self) -> isize { 10 } }
impl Bar for A { fn g(&self) -> isize { 20 } }
impl Baz for A { fn h(&self) -> isize { 30 } }

// Call a function on Foo, given a T: Baz,
// which is inherited via Bar
fn gg<T:Baz>(a: &T) -> isize {
    a.f()
}

pub fn main() {
    let a = &A { x: 3 };
    assert_eq!(gg(a), 10);
}
