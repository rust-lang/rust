#![feature(extern_types)]

extern "C" {
    type A;
    type B;
}

fn foo(r: &A) -> &B {
    r //~ ERROR mismatched types
}

fn main() {}
