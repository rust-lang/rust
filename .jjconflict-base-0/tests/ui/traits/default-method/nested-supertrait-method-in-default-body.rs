//! Regression test for <https://github.com/rust-lang/rust/issues/3979>.
//! Test calling nested supertrait's method in default method body works.
//@ check-pass

trait A {
    fn a_method(&self);
}

trait B: A {
    fn b_method(&self);
}

trait C: B {
    fn c_method(&self) {
        self.a_method();
    }
}

pub fn main() {}
