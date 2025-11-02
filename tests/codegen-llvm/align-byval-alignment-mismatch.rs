// ignore-tidy-linelength
//@ add-core-stubs
//@ revisions:i686-linux x86_64-linux

//@ compile-flags: -Cno-prepopulate-passes -Copt-level=1 -Cpanic=abort
//@[i686-linux] compile-flags: --target i686-unknown-linux-gnu
//@[i686-linux] needs-llvm-components: x86
//@[x86_64-linux] compile-flags: --target x86_64-unknown-linux-gnu
//@[x86_64-linux] needs-llvm-components: x86

// Tests that we correctly copy arguments into allocas when the alignment of the byval argument
// is different from the alignment of the Rust type.

// For the following test cases:
// All of the `*_decreases_alignment` functions should codegen to a direct call, since the
// alignment is already sufficient.
// All off the `*_increases_alignment` functions should copy the argument to an alloca
// on i686-unknown-linux-gnu, since the alignment needs to be increased, and should codegen
// to a direct call on x86_64-unknown-linux-gnu, where byval alignment matches Rust alignment.

#![feature(no_core)]
#![crate_type = "lib"]
#![no_std]
#![no_core]
#![allow(non_camel_case_types)]

extern crate minicore;
use minicore::*;

// This type has align 1 in Rust, but as a byval argument on i686-linux, it will have align 4.
#[repr(C)]
#[repr(packed)]
struct Align1 {
    x: u128,
    y: u128,
    z: u128,
}

// This type has align 16 in Rust, but as a byval argument on i686-linux, it will have align 4.
#[repr(C)]
#[repr(align(16))]
struct Align16 {
    x: u128,
    y: u128,
    z: u128,
}

extern "C" {
    fn extern_c_align1(x: Align1);
    fn extern_c_align16(x: Align16);
}

// CHECK-LABEL: @rust_to_c_increases_alignment
#[no_mangle]
pub unsafe fn rust_to_c_increases_alignment(x: Align1) {
    // i686-linux: start:
    // i686-linux-NEXT: [[ALLOCA:%[0-9a-z]+]] = alloca [48 x i8], align 4
    // i686-linux-NEXT: call void @llvm.lifetime.start.p0({{(i64 48, )?}}ptr {{.*}}[[ALLOCA]])
    // i686-linux-NEXT: call void @llvm.memcpy.{{.+}}(ptr {{.*}}align 4 {{.*}}[[ALLOCA]], ptr {{.*}}align 1 {{.*}}%x
    // i686-linux-NEXT: call void @extern_c_align1({{.+}} [[ALLOCA]])
    // i686-linux-NEXT: call void @llvm.lifetime.end.p0({{(i64 48, )?}}ptr {{.*}}[[ALLOCA]])

    // x86_64-linux: start:
    // x86_64-linux-NEXT: call void @extern_c_align1
    extern_c_align1(x);
}

// CHECK-LABEL: @rust_to_c_decreases_alignment
#[no_mangle]
pub unsafe fn rust_to_c_decreases_alignment(x: Align16) {
    // CHECK: start:
    // CHECK-NEXT: call void @extern_c_align16
    extern_c_align16(x);
}

extern "Rust" {
    fn extern_rust_align1(x: Align1);
    fn extern_rust_align16(x: Align16);
}

// CHECK-LABEL: @c_to_rust_decreases_alignment
#[no_mangle]
pub unsafe extern "C" fn c_to_rust_decreases_alignment(x: Align1) {
    // CHECK: start:
    // CHECK-NEXT: call void @extern_rust_align1
    extern_rust_align1(x);
}

// CHECK-LABEL: @c_to_rust_increases_alignment
#[no_mangle]
pub unsafe extern "C" fn c_to_rust_increases_alignment(x: Align16) {
    // i686-linux: start:
    // i686-linux-NEXT: [[ALLOCA:%[0-9a-z]+]] = alloca [48 x i8], align 16
    // i686-linux-NEXT: call void @llvm.memcpy.{{.+}}(ptr {{.*}}align 16 {{.*}}[[ALLOCA]], ptr {{.*}}align 4 {{.*}}%0
    // i686-linux-NEXT: call void @extern_rust_align16({{.+}} [[ALLOCA]])

    // x86_64-linux: start:
    // x86_64-linux-NEXT: call void @extern_rust_align16
    extern_rust_align16(x);
}

extern "Rust" {
    fn extern_rust_ref_align1(x: &Align1);
    fn extern_rust_ref_align16(x: &Align16);
}

// CHECK-LABEL: @c_to_rust_ref_decreases_alignment
#[no_mangle]
pub unsafe extern "C" fn c_to_rust_ref_decreases_alignment(x: Align1) {
    // CHECK: start:
    // CHECK-NEXT: call void @extern_rust_ref_align1
    extern_rust_ref_align1(&x);
}

// CHECK-LABEL: @c_to_rust_ref_increases_alignment
#[no_mangle]
pub unsafe extern "C" fn c_to_rust_ref_increases_alignment(x: Align16) {
    // i686-linux: start:
    // i686-linux-NEXT: [[ALLOCA:%[0-9a-z]+]] = alloca [48 x i8], align 16
    // i686-linux-NEXT: call void @llvm.memcpy.{{.+}}(ptr {{.*}}align 16 {{.*}}[[ALLOCA]], ptr {{.*}}align 4 {{.*}}%0
    // i686-linux-NEXT: call void @extern_rust_ref_align16({{.+}} [[ALLOCA]])

    // x86_64-linux: start:
    // x86_64-linux-NEXT: call void @extern_rust_ref_align16
    extern_rust_ref_align16(&x);
}
