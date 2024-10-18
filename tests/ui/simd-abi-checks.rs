//@ only-x86_64
//@ build-fail

#![feature(avx512_target_feature)]
#![feature(portable_simd)]
#![allow(improper_ctypes_definitions)]

use std::arch::x86_64::*;

#[repr(transparent)]
struct Wrapper(__m256);

unsafe extern "C" fn w(_: Wrapper) {
    //~^ ABI error: this function definition uses a avx vector type, which is not enabled
    todo!()
}

unsafe extern "C" fn f(_: __m256) {
    //~^ ABI error: this function definition uses a avx vector type, which is not enabled
    todo!()
}

unsafe extern "C" fn g() -> __m256 {
    //~^ ABI error: this function definition uses a avx vector type, which is not enabled
    todo!()
}

#[target_feature(enable = "avx2")]
unsafe extern "C" fn favx(_: __m256) {
    todo!()
}

#[target_feature(enable = "avx")]
unsafe extern "C" fn gavx() -> __m256 {
    todo!()
}

fn as_f64x8(d: __m512d) -> std::simd::f64x8 {
    unsafe { std::mem::transmute(d) }
}

unsafe fn test() {
    let arg = std::mem::transmute([0.0f64; 8]);
    as_f64x8(arg);
}

fn main() {
    unsafe {
        f(g());
        //~^ ERROR ABI error: this function call uses a avx vector type, which is not enabled in the caller
        //~| ERROR ABI error: this function call uses a avx vector type, which is not enabled in the caller
    }

    unsafe {
        favx(gavx());
        //~^ ERROR ABI error: this function call uses a avx vector type, which is not enabled in the caller
        //~| ERROR ABI error: this function call uses a avx vector type, which is not enabled in the caller
    }

    unsafe {
        test();
    }

    unsafe {
        w(Wrapper(g()));
        //~^ ERROR ABI error: this function call uses a avx vector type, which is not enabled in the caller
        //~| ERROR ABI error: this function call uses a avx vector type, which is not enabled in the caller
    }
}
