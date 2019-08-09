// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
struct A<'a> {
    a: &'a i32,
    b: &'a i32,
}

impl <'a> A<'a> {
    fn foo<'b>(&'b self) {
        A {
            a: self.a,
            b: self.b,
        };
    }
}

fn main() { }
