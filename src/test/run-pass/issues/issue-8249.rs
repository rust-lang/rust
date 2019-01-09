// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

trait A {
    fn dummy(&self) { }
}
struct B;
impl A for B {}

struct C<'a> {
    foo: &'a mut (A+'a),
}

fn foo(a: &mut A) {
    C{ foo: a };
}

pub fn main() {
}
