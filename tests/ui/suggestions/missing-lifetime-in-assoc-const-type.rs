//@ check-pass

//@ revisions: default generic_const_items

#![cfg_attr(generic_const_items, feature(generic_const_items), allow(incomplete_features))]

trait ZstAssert: Sized {
    const A: &str = "";
    const B: S = S { s: &() };
    const C: &'_ str = "";
    const D: T = T { a: &(), b: &() };
}

const A: &str = "";
const B: S = S { s: &() };
const C: &'_ str = "";
const D: T = T { a: &(), b: &() };

struct S<'a> {
    s: &'a (),
}
struct T<'a, 'b> {
    a: &'a (),
    b: &'b (),
}

fn main() {}
