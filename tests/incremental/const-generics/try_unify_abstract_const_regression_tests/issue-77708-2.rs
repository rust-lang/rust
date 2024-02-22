//@ revisions: rpass
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

struct Z;
const fn one() -> usize {
    1
}

fn from_a_to_b<T>(source: [u8; one()]) -> T {
    todo!()
}

fn not_main() {
    let _: &Z = from_a_to_b([0; 1]);
}

fn main() {}
