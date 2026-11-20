#![feature(adt_const_params, const_param_ty_trait, generic_const_parameter_types)]
#![allow(incomplete_features)]

use std::marker::ConstParamTy_;

fn foo<const N: usize, const M: [u8; N]>() {}
fn bar<T: ConstParamTy_, const M: [T; 2]>() {}

fn main() {
    foo::<3, { [1; 2] }>();
    //~^ ERROR: mismatched type

    bar::<u8, { [2_u16; 2] }>();
    //~^ ERROR: mismatched type
}
