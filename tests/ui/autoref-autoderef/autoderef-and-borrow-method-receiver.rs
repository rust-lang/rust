//@ run-pass
#![allow(dead_code)]

struct Foo {
    x: isize,
}

impl Foo {
    pub fn f(&self) {}
}

fn g(x: &mut Foo) {
    x.f();
}

pub fn main() {
}
