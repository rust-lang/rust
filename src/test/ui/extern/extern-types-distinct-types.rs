#![feature(extern_types)]

extern {
    type A;
    type B;
}

fn foo(r: &A) -> &B {
    r //~ ERROR mismatched types
}

fn main() { }
