//@ check-pass

#![feature(adt_const_params, const_param_ty_trait, generic_const_parameter_types)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy_;

fn foo<T: ConstParamTy_, const N: usize, const M: [T; N]>() -> [T; N] {
    M
}

fn main() {
    let a: [u8; 2] = foo::<u8, 2, { [12; _] }>();
}
