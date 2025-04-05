//@ only-x86_64
//@ build-fail
//@ compile-flags: -C target-feature=-avx

#![feature(avx512_target_feature)]
#![feature(portable_simd)]
#![feature(simd_ffi)]
#![allow(improper_ctypes_definitions)]

use std::arch::x86_64::*;

#[repr(transparent)]
struct Wrapper(__m256);

unsafe extern "C" fn w(_: Wrapper) {
    //~^ ERROR: requires the `avx` target feature, which is not enabled
    todo!()
}

unsafe extern "C" fn f(_: __m256) {
    //~^ ERROR: requires the `avx` target feature, which is not enabled
    todo!()
}

unsafe extern "C" fn g() -> __m256 {
    //~^ ERROR: requires the `avx` target feature, which is not enabled
    todo!()
}

#[target_feature(enable = "avx")]
unsafe extern "C" fn favx() -> __m256 {
    todo!()
}

// avx2 implies avx, so no error here.
#[target_feature(enable = "avx2")]
unsafe extern "C" fn gavx(_: __m256) {
    todo!()
}

// No error because of "Rust" ABI.
fn as_f64x8(d: __m512d) -> std::simd::f64x8 {
    unsafe { std::mem::transmute(d) }
}

unsafe fn test() {
    let arg = std::mem::transmute([0.0f64; 8]);
    as_f64x8(arg);
}

#[target_feature(enable = "avx")]
unsafe fn in_closure() -> impl FnOnce() -> __m256 {
    #[inline(always)] // this disables target-feature inheritance
    || g()
    //~^ ERROR requires the `avx` target feature, which is not enabled in the caller
}

fn main() {
    unsafe {
        f(g());
        //~^ ERROR requires the `avx` target feature, which is not enabled in the caller
        //~| ERROR requires the `avx` target feature, which is not enabled in the caller
    }

    unsafe {
        gavx(favx());
        //~^ ERROR requires the `avx` target feature, which is not enabled in the caller
        //~| ERROR requires the `avx` target feature, which is not enabled in the caller
    }

    unsafe {
        test();
    }

    unsafe {
        w(Wrapper(g()));
        //~^ ERROR requires the `avx` target feature, which is not enabled in the caller
        //~| ERROR requires the `avx` target feature, which is not enabled in the caller
    }

    unsafe {
        in_closure()();
    }

    unsafe {
        #[expect(improper_ctypes)]
        extern "C" {
            fn some_extern() -> __m256;
        }
        some_extern();
        //~^ ERROR requires the `avx` target feature, which is not enabled in the caller
    }
}

#[no_mangle]
#[target_feature(enable = "avx")]
fn some_extern() -> __m256 {
    todo!()
}
