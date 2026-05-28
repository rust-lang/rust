//@ run-pass
#![expect(incomplete_features)]
#![feature(min_generic_const_args, adt_const_params)]
#![allow(dead_code)]

macro_rules! make_array {
    ($n:expr, $m:expr, $p:expr) => {
        [N, $m, $p]
    };
}

fn takes_array<const A: [u32; 3]>() {}

fn generic_caller<const N: u32>() {
    takes_array::<{ make_array!(N, 2, 3) }>();
}

fn main() {}
