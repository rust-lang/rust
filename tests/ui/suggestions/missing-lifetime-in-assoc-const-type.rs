// revisions: default generic_const_items

#![cfg_attr(generic_const_items, feature(generic_const_items), allow(incomplete_features))]

trait ZstAssert: Sized {
    const A: &str = ""; //~ ERROR missing lifetime specifier
    const B: S = S { s: &() }; //~ ERROR missing lifetime specifier
    const C: &'_ str = ""; //~ ERROR missing lifetime specifier
    const D: T = T { a: &(), b: &() }; //~ ERROR missing lifetime specifier
}

struct S<'a> {
    s: &'a (),
}
struct T<'a, 'b> {
    a: &'a (),
    b: &'b (),
}

fn main() {}
