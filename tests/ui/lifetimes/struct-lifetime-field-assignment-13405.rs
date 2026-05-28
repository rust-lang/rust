//! Regression test for https://github.com/rust-lang/rust/issues/13405

//@ check-pass
#![allow(dead_code)]
#![allow(unused_variables)]

struct Foo<'a> {
    i: &'a bool,
    j: Option<&'a isize>,
}

impl<'a> Foo<'a> {
    fn bar(&mut self, j: &isize) {
        let child = Foo {
            i: self.i,
            j: Some(j)
        };
    }
}

fn main() {}
