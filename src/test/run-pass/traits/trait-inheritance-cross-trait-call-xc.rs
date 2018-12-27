// run-pass
// aux-build:trait_xc_call_aux.rs


extern crate trait_xc_call_aux as aux;

use aux::Foo;

trait Bar : Foo {
    fn g(&self) -> isize;
}

impl Bar for aux::A {
    fn g(&self) -> isize { self.f() }
}

pub fn main() {
    let a = &aux::A { x: 3 };
    assert_eq!(a.g(), 10);
}
