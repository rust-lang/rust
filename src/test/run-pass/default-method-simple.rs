#![allow(dead_code)]

trait Foo {
    fn f(&self) {
        println!("Hello!");
        self.g();
    }
    fn g(&self);
}

struct A {
    x: isize
}

impl Foo for A {
    fn g(&self) {
        println!("Goodbye!");
    }
}

pub fn main() {
    let a = A { x: 1 };
    a.f();
}
