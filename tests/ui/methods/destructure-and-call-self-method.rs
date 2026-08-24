//! Regression test for <https://github.com/rust-lang/rust/issues/4759>.
//! Destructuring a struct and calling consuming method used to segfault.
//@ run-pass

#![allow(non_shorthand_field_patterns)]

struct T { a: Box<isize> }

trait U {
    fn f(self);
}

impl U for Box<isize> {
    fn f(self) { }
}

pub fn main() {
    let T { a: a } = T { a: Box::new(0) };
    a.f();
}
