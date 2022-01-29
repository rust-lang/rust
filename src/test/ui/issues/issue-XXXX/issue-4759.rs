// run-pass
// pretty-expanded FIXME #23616
#![allow(non_shorthand_field_patterns)]

#![feature(box_syntax)]

struct T { a: Box<isize> }

trait U {
    fn f(self);
}

impl U for Box<isize> {
    fn f(self) { }
}

pub fn main() {
    let T { a: a } = T { a: box 0 };
    a.f();
}
