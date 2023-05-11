// run-pass
#![allow(dead_code)]
// pretty-expanded FIXME #23616

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
