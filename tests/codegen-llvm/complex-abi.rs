//! Checks that `#[repr(complex)]` `Complex<T>` matches the C `_Complex` ABI in `extern "C"`
//! functions. This is the rustc side of `tests/run-make/complex-c-abi`, which additionally
//! checks these signatures against clang. Revisions are grouped by LLVM component.

//@ add-minicore
//@ compile-flags: -C no-prepopulate-passes -Z codegen-source-order -C opt-level=3

#![feature(no_core, lang_items, repr_complex, f16, f128)]
#![no_core]
#![allow(improper_ctypes)] // only Complex<{float}> is guaranteed to be ABI-compatible for now
#![crate_type = "lib"]

extern crate minicore;
use minicore::num::Complex;

#[no_mangle]
pub extern "C" fn cplx_f16(x: Complex<f16>) -> Complex<f16> {
    // CHECK: cplx_f16
    x
}

#[no_mangle]
pub extern "C" fn cplx_f32(x: Complex<f32>) -> Complex<f32> {
    x
}

#[no_mangle]
pub extern "C" fn cplx_f64(x: Complex<f64>) -> Complex<f64> {
    x
}

#[no_mangle]
pub extern "C" fn cplx_f128(x: Complex<f128>) -> Complex<f128> {
    x
}

#[no_mangle]
pub extern "C" fn cplx_i8(x: Complex<i8>) -> Complex<i8> {
    x
}

#[no_mangle]
pub extern "C" fn cplx_i16(x: Complex<i16>) -> Complex<i16> {
    x
}

#[no_mangle]
pub extern "C" fn cplx_i32(x: Complex<i32>) -> Complex<i32> {
    x
}

#[no_mangle]
pub extern "C" fn cplx_i64(x: Complex<i64>) -> Complex<i64> {
    x
}

#[repr(transparent)]
struct Wrapper<T>(T);

#[no_mangle]
pub extern "C" fn wrapper_cplx_i64(x: Wrapper<Complex<i64>>) -> Wrapper<Complex<i64>> {
    // I686:           define{{.*}} void @wrapper_cplx_i64(ptr {{.*}} sret([16 x i8]) {{.*}}, ptr {{.*}} byval([16 x i8]) {{.*}})
    // WIN32_GNU:      define{{.*}} void @wrapper_cplx_i64(ptr {{.*}} sret([16 x i8]) {{.*}}, ptr {{.*}} byval([16 x i8]) {{.*}})
    // WIN32_MSVC:     define{{.*}} void @wrapper_cplx_i64(ptr {{.*}} sret([16 x i8]) {{.*}}, ptr {{.*}} byval([16 x i8]) {{.*}})
    // WINDOWS_GNU:    define{{.*}} void @wrapper_cplx_i64(ptr {{.*}} sret([16 x i8]) {{.*}}, ptr {{.*}})
    // WINDOWS_MSVC:   define{{.*}} void @wrapper_cplx_i64(ptr {{.*}} sret([16 x i8]) {{.*}}, ptr {{.*}})
    // X86_64:         define{{.*}} { i64, i64 } @wrapper_cplx_i64({ i64, i64 } {{.*}})
    x
}
