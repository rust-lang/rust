//@ build-pass
//@ compile-flags: --crate-type=lib
#![feature(f16b, repr_simd)]

extern crate core;

#[allow(non_camel_case_types)]
type bfloat16_t = core::num::f16b;

#[repr(simd)]
#[allow(non_camel_case_types)]
pub struct bfloat16x4([bfloat16_t; 4]);
