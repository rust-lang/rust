//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]
#![feature(portable_simd)]

// Regression test for https://github.com/rust-lang/rust/issues/97804.

use std::mem;
use std::simd::u8x16;

#[unsafe(no_mangle)]
fn foo(v: u16, p: &mut [u8; 16]) {
    // An array repeat transmuted into a SIMD type should emit a canonical LLVM splat sequence:
    //
    // CHECK-LABEL: foo
    // CHECK: start
    // CHECK-NEXT: %0 = insertelement <8 x i16> poison, i16 %v, i64 0
    // CHECK-NEXT: %1 = shufflevector <8 x i16> %0, <8 x i16> poison, <8 x i32> zeroinitializer
    // CHECK-NEXT: store <8 x i16> %1, ptr %p, align 1
    // CHECK-NEXT: ret void
    unsafe {
        let v: u8x16 = mem::transmute([v; 8]);
        *p = mem::transmute(v);
    }
}
