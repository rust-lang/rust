//! Ensure we don't allow Vec<[u8]> as const parameter even with
//! `const_param_ty_unchecked` feature.
#![allow(incomplete_features)]
#![feature(adt_const_params, const_param_ty_unchecked, const_param_ty_trait)]
use std::marker::ConstParamTy_;

struct VectorOfBytes {
    a: Vec<[u8]>
    //~^ ERROR: the size for values of type `[u8]` cannot be known at compilation time [E0277]
}
impl ConstParamTy_ for VectorOfBytes {}

fn bar<const N: VectorOfBytes>() {}
fn foo<const N: Vec<[u8]>>() {}
    //~^ ERROR: the size for values of type `[u8]` cannot be known at compilation time [E0277]
    //~| ERROR: the size for values of type `[u8]` cannot be known at compilation time [E0277]


fn main() {}
