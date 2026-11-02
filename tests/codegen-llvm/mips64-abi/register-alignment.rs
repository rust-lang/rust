//@ add-minicore
//@ compile-flags: -Copt-level=3
//
//@ revisions: mips64 mips64el
//@[mips64] compile-flags: --target mips64-unknown-linux-gnuabi64
//@[mips64el] compile-flags: --target mips64el-unknown-linux-gnuabi64
//
//@ needs-llvm-components: mips

// Test that 16-byte scalars are correctly aligned in registers.

#![feature(f128, lang_items, no_core)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;

// CHECK: @i128_gets_aligned(i8 %0, i32 %1, i128 %2)
#[no_mangle]
extern "C" fn i128_gets_aligned(arg0: u8, arg1: i128) {}

// CHECK: @i128_is_aligned(i8 %0, i16 %1, i128 %2)
#[no_mangle]
extern "C" fn i128_is_aligned(arg0: u8, arg1: i16, arg2: i128) {}

// CHECK: @f128_gets_aligned(i8 %0, i32 %1, fp128 %2)
#[no_mangle]
extern "C" fn f128_gets_aligned(arg0: u8, arg1: f128) {}

// CHECK: @f128_is_aligned(i8 %0, float %1, fp128 %2)
#[no_mangle]
extern "C" fn f128_is_aligned(arg0: u8, arg1: f32, arg2: f128) {}
