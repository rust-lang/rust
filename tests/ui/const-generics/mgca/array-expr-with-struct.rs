//@ run-pass
#![feature(min_generic_const_args, adt_const_params)]
#![expect(incomplete_features)]
#![allow(dead_code)]

use std::marker::ConstParamTy;

#[derive(PartialEq, Eq, ConstParamTy)]
struct Container {
    values: [u32; 3],
}

fn takes_container<const C: Container>() {}

fn generic_caller<const N: u32, const M: u32>() {
    takes_container::<{ Container { values: [N, M, 1] } }>();
    takes_container::<{ Container { values: [1, 2, 3] } }>();
}

fn main() {}
