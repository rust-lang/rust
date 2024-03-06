// Test that unsized locals uphold alignment requirements.
// Regression test for #71416.
//@ run-pass
#![feature(unsized_locals)]
#![allow(incomplete_features)]
use std::any::Any;

#[repr(align(256))]
#[allow(dead_code)]
struct A {
    v: u8
}

impl A {
    fn f(&self) -> *const A {
        assert_eq!(self as *const A as usize % 256, 0);
        self
    }
}

fn mk() -> Box<dyn Any> {
    Box::new(A { v: 4 })
}

fn main() {
    let x = *mk();
    let dwncst = x.downcast_ref::<A>().unwrap();
    let addr = dwncst.f();
    assert_eq!(addr as usize % 256, 0);
}
