// https://github.com/rust-lang/rust/issues/8249
//@ run-pass
#![allow(dead_code)]

trait A {
    fn dummy(&self) { }
}
struct B;
impl A for B {}

struct C<'a> {
    foo: &'a mut (dyn A+'a),
}

fn foo(a: &mut dyn A) {
    C{ foo: a };
}

pub fn main() {
}
