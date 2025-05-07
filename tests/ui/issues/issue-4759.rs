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
