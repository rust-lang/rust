//@ compile-flags: -Copt-level=3 -Ctarget-feature=+simd128
//@ min-llvm-version: 23

#![feature(portable_simd)]
#![no_std]
#![crate_type = "rlib"]
use core::simd::prelude::*;

// CHECK-LABEL: @a
// CHECK: start:
// CHECK-NEXT: insertelement <8 x i16> poison, i16 %b, i64 0
// CHECK-NEXT: shufflevector <8 x i16> %0, <8 x i16> poison, <8 x i32> zeroinitializer
// CHECK-NEXT: icmp eq <8 x i16> %a, %_3
// CHECK-NEXT: bitcast <8 x i1> %1 to i8
// CHECK-NEXT: icmp ne i8 %2, 0
// CHECK-NEXT: ret i1 %_0

#[no_mangle]
fn a(a: u16x8, b: u16) -> bool {
    a.simd_eq(Simd::splat(b)).any()
}
// CHECK-LABEL: @b
// CHECK: start:
// CHECK-NEXT: insertelement <16 x i16> poison, i16 %b, i64 0
// CHECK-NEXT: shufflevector <16 x i16> %0, <16 x i16> poison, <16 x i32> zeroinitializer
// CHECK-NEXT: load <16 x i16>, ptr %a, align 32
// CHECK-NEXT: icmp eq <16 x i16> %1, %_3
// CHECK-NEXT: bitcast <16 x i1> %2 to i16
// CHECK-NEXT: icmp ne i16 %3, 0
// CHECK-NEXT: ret i1 %_0

#[no_mangle]
fn b(a: u16x16, b: u16) -> bool {
    a.simd_eq(Simd::splat(b)).any()
}

// CHECK-LABEL: @c
// CHECK: start:
// CHECK-NEXT: insertelement <32 x i16> poison, i16 %b, i64 0
// CHECK-NEXT: shufflevector <32 x i16> %0, <32 x i16> poison, <32 x i32> zeroinitializer
// CHECK-NEXT: load <32 x i16>, ptr %a, align 64
// CHECK-NEXT: icmp eq <32 x i16> %1, %_3
// CHECK-NEXT: bitcast <32 x i1> %2 to i32
// CHECK-NEXT: icmp ne i32 %3, 0
// CHECK-NEXT: ret i1 %_0

#[no_mangle]
fn c(a: u16x32, b: u16) -> bool {
    a.simd_eq(Simd::splat(b)).any()
}
