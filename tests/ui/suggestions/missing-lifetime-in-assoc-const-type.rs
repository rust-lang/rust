//@ revisions: default generic_const_items

#![cfg_attr(generic_const_items, feature(generic_const_items), allow(incomplete_features))]

trait ZstAssert: Sized {
    const A: &str = "";
    const B: S = S { s: &() }; //~ ERROR implicit elided lifetime not allowed here
    const C: &'_ str = "";
    const D: T = T { a: &(), b: &() }; //~ ERROR implicit elided lifetime not allowed here
}

struct S<'a> {
    s: &'a (),
}
struct T<'a, 'b> {
    a: &'a (),
    b: &'b (),
}

fn main() {}
