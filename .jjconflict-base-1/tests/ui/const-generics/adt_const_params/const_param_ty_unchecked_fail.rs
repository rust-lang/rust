// gate-test-const_param_ty_unchecked
//! Ensure this fails when const_param_ty_unchecked isn't used
#![allow(incomplete_features)]
#![feature(const_param_ty_trait)]

use std::marker::ConstParamTy_;

struct Miow;

struct Meoww(Miow);

struct Float {
    float: f32,
}

impl ConstParamTy_ for Meoww {}
                    //~^ ERROR: the trait `ConstParamTy_` cannot be implemented for this type [E0204]
impl ConstParamTy_ for Float {}
                    //~^ ERROR: the trait `ConstParamTy_` cannot be implemented for this type [E0204]

fn something2<const N: *mut u8>() {}
                   //~^ ERROR: using raw pointers as const generic parameters is forbidden
fn something<const N: f64>(a: f64) -> f64 {
                   //~^ ERROR: `f64` is forbidden as the type of a const generic parameter
    N + a
}
fn foo<const N: Vec<[u8]>>() {}
    //~^  ERROR: the size for values of type `[u8]` cannot be known at compilation time [E0277]
          //~^^ ERROR: `Vec<[u8]>` is forbidden as the type of a const generic parameter

fn main() {}
