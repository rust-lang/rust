//@ compile-flags: -Copt-level=3 -Ctarget-feature=+simd128
//@ min-llvm-version: 23

#![feature(portable_simd)]
#![no_std]
#![crate_type = "rlib"]
use core::simd::prelude::*;

// CHECK-LABEL: @a
// CHECK: start:
// CHECK-NEXT: [[INS:%.*]] = insertelement <8 x i16> poison, i16 %b, i64 0
// CHECK-NEXT: [[SPLAT:%.*]] = shufflevector <8 x i16> [[INS]], <8 x i16> poison, <8 x i32> zeroinitializer
// CHECK-NEXT: [[CMP:%.*]] = icmp eq <8 x i16> %a, [[SPLAT]]
// CHECK-NEXT: [[BCST:%.*]] = bitcast <8 x i1> [[CMP]] to i8
// CHECK-NEXT: [[CMPNE:%.*]] = icmp ne i8 [[BCST]], 0
// CHECK-NEXT: ret i1 [[CMPNE]]

#[no_mangle]
fn a(a: u16x8, b: u16) -> bool {
    a.simd_eq(Simd::splat(b)).any()
}
// CHECK-LABEL: @b
// CHECK: start:
// CHECK-NEXT: [[INS:%.*]] = insertelement <16 x i16> poison, i16 %b, i64 0
// CHECK-NEXT: [[SPLAT:%.*]] = shufflevector <16 x i16> [[INS]], <16 x i16> poison, <16 x i32> zeroinitializer
// CHECK-NEXT: [[LOAD:%.*]] = load <16 x i16>, ptr %a, align 32
// CHECK-NEXT: [[CMP:%.*]] = icmp eq <16 x i16> [[LOAD]], [[SPLAT]] 
// CHECK-NEXT: [[BCST:%.*]] = bitcast <16 x i1> [[CMP]] to i16
// CHECK-NEXT: [[CMPNE:%.*]] = icmp ne i16 [[BCST]], 0
// CHECK-NEXT: ret i1 [[CMPNE]]

#[no_mangle]
fn b(a: u16x16, b: u16) -> bool {
    a.simd_eq(Simd::splat(b)).any()
}

// CHECK-LABEL: @c
// CHECK: start:
// CHECK-NEXT: [[INS:%.*]] = insertelement <32 x i16> poison, i16 %b, i64 0
// CHECK-NEXT: [[SPLAT:%.*]] = shufflevector <32 x i16> [[INS]], <32 x i16> poison, <32 x i32> zeroinitializer
// CHECK-NEXT: [[LOAD:%.*]] = load <32 x i16>, ptr %a, align 64
// CHECK-NEXT: [[CMP:%.*]] = icmp eq <32 x i16> [[LOAD]], [[SPLAT]]
// CHECK-NEXT: [[BCST:%.*]] = bitcast <32 x i1> [[CMP]] to i32
// CHECK-NEXT: [[CMPNE:%.*]] = icmp ne i32 [[BCST]], 0
// CHECK-NEXT: ret i1 [[CMPNE]]

#[no_mangle]
fn c(a: u16x32, b: u16) -> bool {
    a.simd_eq(Simd::splat(b)).any()
}
