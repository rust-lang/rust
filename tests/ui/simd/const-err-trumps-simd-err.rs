//@build-fail
//@ dont-require-annotations: NOTE

//! Make sure that monomorphization-time const errors from `static_assert` take priority over the
//! error from simd_extract. Basically this checks that if a const fails to evaluate in some
//! function, we don't bother codegen'ing the function.
#![feature(core_intrinsics)]
#![feature(repr_simd)]

use std::intrinsics::simd::*;

#[repr(simd)]
#[allow(non_camel_case_types)]
struct int8x4_t([u8; 4]);

fn get_elem<const LANE: u32>(a: int8x4_t) -> u8 {
    const { assert!(LANE < 4); } // the error should be here...
    //~^ ERROR assertion failed
    unsafe { simd_extract(a, LANE) } // ...not here
}

fn main() {
    get_elem::<4>(int8x4_t([0, 0, 0, 0]));
}
