// run-pass
#![allow(dead_code)]
// aux-build:trait_inheritance_auto_xc_aux.rs


extern crate trait_inheritance_auto_xc_aux as aux;

use aux::{Foo, Bar, Baz, Quux};

struct A { x: isize }

impl Foo for A { fn f(&self) -> isize { 10 } }
impl Bar for A { fn g(&self) -> isize { 20 } }
impl Baz for A { fn h(&self) -> isize { 30 } }

fn f<T:Quux>(a: &T) {
    assert_eq!(a.f(), 10);
    assert_eq!(a.g(), 20);
    assert_eq!(a.h(), 30);
}

pub fn main() {
    let a = &A { x: 3 };
    f(a);
}
