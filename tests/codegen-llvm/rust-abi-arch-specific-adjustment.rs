//@ add-core-stubs
//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes
//@ revisions: riscv64 loongarch64

//@[riscv64] compile-flags: --target riscv64gc-unknown-linux-gnu
//@[riscv64] needs-llvm-components: riscv

//@[loongarch64] compile-flags: --target loongarch64-unknown-linux-gnu
//@[loongarch64] needs-llvm-components: loongarch

#![crate_type = "lib"]
#![feature(no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

#[no_mangle]
// riscv64:     define noundef i8 @arg_attr_u8(i8 noundef zeroext %x)
// loongarch64: define noundef i8 @arg_attr_u8(i8 noundef zeroext %x)
pub fn arg_attr_u8(x: u8) -> u8 {
    x
}

#[no_mangle]
// riscv64:     define noundef i16 @arg_attr_u16(i16 noundef zeroext %x)
// loongarch64: define noundef i16 @arg_attr_u16(i16 noundef zeroext %x)
pub fn arg_attr_u16(x: u16) -> u16 {
    x
}

#[no_mangle]
// riscv64:     define noundef i32 @arg_attr_u32(i32 noundef signext %x)
// loongarch64: define noundef i32 @arg_attr_u32(i32 noundef signext %x)
pub fn arg_attr_u32(x: u32) -> u32 {
    x
}

#[no_mangle]
// riscv64:     define noundef i64 @arg_attr_u64(i64 noundef %x)
// loongarch64: define noundef i64 @arg_attr_u64(i64 noundef %x)
pub fn arg_attr_u64(x: u64) -> u64 {
    x
}

#[no_mangle]
// riscv64:     define noundef i128 @arg_attr_u128(i128 noundef %x)
// loongarch64: define noundef i128 @arg_attr_u128(i128 noundef %x)
pub fn arg_attr_u128(x: u128) -> u128 {
    x
}

#[no_mangle]
// riscv64:     define noundef i8 @arg_attr_i8(i8 noundef signext %x)
// loongarch64: define noundef i8 @arg_attr_i8(i8 noundef signext %x)
pub fn arg_attr_i8(x: i8) -> i8 {
    x
}

#[no_mangle]
// riscv64:     define noundef i16 @arg_attr_i16(i16 noundef signext %x)
// loongarch64: define noundef i16 @arg_attr_i16(i16 noundef signext %x)
pub fn arg_attr_i16(x: i16) -> i16 {
    x
}

#[no_mangle]
// riscv64:     define noundef i32 @arg_attr_i32(i32 noundef signext %x)
// loongarch64: define noundef i32 @arg_attr_i32(i32 noundef signext %x)
pub fn arg_attr_i32(x: i32) -> i32 {
    x
}

#[no_mangle]
// riscv64:     define noundef i64 @arg_attr_i64(i64 noundef %x)
// loongarch64: define noundef i64 @arg_attr_i64(i64 noundef %x)
pub fn arg_attr_i64(x: i64) -> i64 {
    x
}

#[no_mangle]
// riscv64:     define noundef i128 @arg_attr_i128(i128 noundef %x)
// loongarch64: define noundef i128 @arg_attr_i128(i128 noundef %x)
pub fn arg_attr_i128(x: i128) -> i128 {
    x
}

#[no_mangle]
// riscv64:     define noundef zeroext i1 @arg_attr_bool(i1 noundef zeroext %x)
// loongarch64: define noundef zeroext i1 @arg_attr_bool(i1 noundef zeroext %x)
pub fn arg_attr_bool(x: bool) -> bool {
    x
}

#[no_mangle]
// ignore-tidy-linelength
// riscv64:     define noundef{{( range\(i32 0, 1114112\))?}} i32 @arg_attr_char(i32 noundef signext{{( range\(i32 0, 1114112\))?}} %x)
// loongarch64: define noundef{{( range\(i32 0, 1114112\))?}} i32 @arg_attr_char(i32 noundef signext{{( range\(i32 0, 1114112\))?}} %x)
pub fn arg_attr_char(x: char) -> char {
    x
}

#[no_mangle]
// riscv64:     define noundef float @arg_attr_f32(float noundef %x)
// loongarch64: define noundef float @arg_attr_f32(float noundef %x)
pub fn arg_attr_f32(x: f32) -> f32 {
    x
}

#[no_mangle]
// riscv64:     define noundef double @arg_attr_f64(double noundef %x)
// loongarch64: define noundef double @arg_attr_f64(double noundef %x)
pub fn arg_attr_f64(x: f64) -> f64 {
    x
}
