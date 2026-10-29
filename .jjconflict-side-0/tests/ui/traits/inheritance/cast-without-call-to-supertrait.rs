//@ run-pass
#![allow(dead_code)]
// Testing that we can cast to a subtrait and call subtrait
// methods. Not testing supertrait methods


trait Foo {
    fn f(&self) -> isize;
}

trait Bar : Foo {
    fn g(&self) -> isize;
}

struct A {
    x: isize
}

impl Foo for A {
    fn f(&self) -> isize { 10 }
}

impl Bar for A {
    fn g(&self) -> isize { 20 }
}

pub fn main() {
    let a = &A { x: 3 };
    let afoo = a as &dyn Foo;
    let abar = a as &dyn Bar;
    assert_eq!(afoo.f(), 10);
    assert_eq!(abar.g(), 20);
}
