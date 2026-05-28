//@ add-minicore
//@ revisions: X86 AARCH64 RISCV S390X
//@ [X86] compile-flags: -Copt-level=3 --target=x86_64-unknown-linux-gnu
//@ [X86] needs-llvm-components: x86
//@ [AARCH64] compile-flags: -Copt-level=3 --target=aarch64-unknown-linux-gnu
//@ [AARCH64] needs-llvm-components: aarch64
//@ [RISCV] compile-flags: -Copt-level=3 --target riscv64gc-unknown-linux-gnu -Ctarget-feature=+v
//@ [RISCV] needs-llvm-components: riscv
//@ [S390X] compile-flags: -Copt-level=3 --target s390x-unknown-linux-gnu -Ctarget-feature=+vector
//@ [S390X] needs-llvm-components: systemz
#![crate_type = "lib"]
#![feature(repr_simd)]
#![feature(no_core)]
#![no_std]
#![no_core]
extern crate minicore;
use minicore::*;

#[repr(simd)]
pub struct Simd<T, const N: usize>(pub [T; N]);

pub type u8x16 = Simd<u8, 16>;

// Regression test for https://github.com/rust-lang/rust/issues/97804.

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
