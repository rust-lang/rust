// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

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
