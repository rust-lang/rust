//@ run-pass
#![allow(dead_code)]

mod foo {
    pub enum Foo {
        Bar { a: isize }
    }
}

fn f(f: foo::Foo) {
    match f {
        foo::Foo::Bar { a: _a } => {}
    }
}

pub fn main() {}
