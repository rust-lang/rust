//@ aux-build:struct_field_privacy.rs

extern crate struct_field_privacy as xc;

struct A {
    a: isize,
}

mod inner {
    pub struct A {
        a: isize,
        pub b: isize,
    }
    pub struct B {
        pub a: isize,
        b: isize,
    }
    pub struct Z(pub isize, isize);
}

fn test(a: A, b: inner::A, c: inner::B, d: xc::A, e: xc::B, z: inner::Z) {
    a.a;
    b.a; //~ ERROR: field `a` of struct `inner::A` is private
    b.b;
    c.a;
    c.b; //~ ERROR: field `b` of struct `inner::B` is private

    d.a; //~ ERROR: field `a` of struct `xc::A` is private
    d.b;

    e.a;
    e.b; //~ ERROR: field `b` of struct `xc::B` is private

    z.0;
    z.1; //~ ERROR: field `1` of struct `Z` is private
}

fn main() {}
