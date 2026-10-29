//! Checks basic default method functionality.

//@ run-pass

trait Foo {
    fn f(&self) {
        println!("Hello!");
        self.g();
    }
    fn g(&self);
}

struct A;

impl Foo for A {
    fn g(&self) {
        println!("Goodbye!");
    }
}

pub fn main() {
    let a = A;
    a.f();
}
