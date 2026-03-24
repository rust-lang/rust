//! This test ensures no errors are emitted when lowering literals with
//! explicitly stated types and inference variables in the type of the const
//! generic parameter.
//!
//! See https://github.com/rust-lang/rust/pull/153557

//@check-pass

#![allow(incomplete_features)]
#![feature(adt_const_params,
    min_generic_const_args,
    generic_const_parameter_types,
    unsized_const_params
)]

use std::marker::ConstParamTy_;

fn main() {
    foo::<_, 2_i32>();
}

fn foo<T: ConstParamTy_, const N: T>() {}
