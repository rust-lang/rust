// check-pass
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(associated_type_defaults)]

fn foo<const A: usize>(a: [u32; A]) -> [u32; A + 1] {
    panic!()
}

const LEN: usize = 1;

fn bar<const N: usize>() {
    foo::<LEN>([123]);
}

fn main() {}
