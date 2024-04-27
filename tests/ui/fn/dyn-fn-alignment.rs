//@ run-pass

#![allow(dead_code)]
#[repr(align(256))]
struct A {
    v: u8,
}

impl A {
    fn f(&self) -> *const A {
        self
    }
}

fn f2(v: u8) -> Box<dyn FnOnce() -> *const A> {
    let a = A { v };
    Box::new(move || a.f())
}

fn main() {
    let addr = f2(0)();
    assert_eq!(addr as usize % 256, 0, "addr: {:?}", addr);
}
