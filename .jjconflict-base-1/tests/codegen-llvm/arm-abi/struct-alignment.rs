//@ add-minicore
//@ compile-flags: -Cno-prepopulate-passes -Copt-level=0
//
//@ revisions: linux eabi watchos
//@[linux] compile-flags: --target armv7-unknown-linux-gnueabihf
//@[eabi] compile-flags: --target armv7r-none-eabi
//@[watchos] compile-flags: --target armv7k-apple-watchos
//
//@ needs-llvm-components: arm

// Test that structs are passed with the correct register alignment.

#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;

#[repr(C)]
pub struct Natural {
    pub a: u32,
    pub b: u32,
}

// linux:   define void @test_natural([2 x i32] %0)
// eabi:    define dso_local void @test_natural([2 x i32] %0)
// watchos: define void @test_natural([2 x i32] %0)
#[unsafe(no_mangle)]
pub extern "C" fn test_natural(a: Natural) {
    hint::black_box(a);
}

#[repr(C)]
pub struct HasU64 {
    pub a: u64,
}

// linux:   define void @test_has_u64(i64 %0)
// eabi:    define dso_local void @test_has_u64(i64 %0)
// watchos: define void @test_has_u64(i64 %0)
#[unsafe(no_mangle)]
pub extern "C" fn test_has_u64(a: HasU64) {
    hint::black_box(a);
}

#[repr(C)]
#[repr(align(8))]
pub struct Align8 {
    pub a: u32,
    pub b: u32,
}

#[repr(transparent)]
pub struct Transparent8 {
    a: Align8,
}

// Here watchOS deviates from the others. On linux and eabi, the natural alignment of the struct
// contents is used, and the alignment modifier of the struct is not considered for ABI purposes.
// watchOS does respect the alignment annotation for register allocation.
//
// linux:   define void @test_8([2 x i32] %0, [2 x i32] %1)
// eabi:    define dso_local void @test_8([2 x i32] %0, [2 x i32] %1)
// watchos: define void @test_8(i64 %0, i64 %1)
#[unsafe(no_mangle)]
pub extern "C" fn test_8(a: Align8, b: Transparent8) {
    hint::black_box(a);
    hint::black_box(b);
}

#[repr(C)]
#[repr(align(16))]
pub struct Align16 {
    pub a: u32,
    pub b: u32,
}

#[repr(transparent)]
pub struct Transparent16 {
    a: Align16,
}

#[repr(C)]
pub struct Wrapped16 {
    pub a: Align16,
}

// Wrapping resets the unadjusted alignment.
//
// linux:   define void @test_16([4 x i32] %0, [4 x i32] %1, [2 x i64] %2)
// eabi:    define dso_local void @test_16([4 x i32] %0, [4 x i32] %1, [2 x i64] %2)
// watchos: define void @test_16([2 x i64] %0, [2 x i64] %1, [2 x i64] %2)
#[unsafe(no_mangle)]
pub extern "C" fn test_16(a: Align16, b: Transparent16, c: Wrapped16) {
    hint::black_box(a);
    hint::black_box(b);
    hint::black_box(c);
}

// Packing is different, the alignment is 1 across targets.
#[repr(C)]
#[repr(packed)]
pub struct Packed {
    pub a: u8,
    pub b: u64,
}

// linux:   define void @test_packed([3 x i32] %0)
// eabi:    define dso_local void @test_packed([3 x i32] %0)
// watchos: define void @test_packed([3 x i32] %0)
#[unsafe(no_mangle)]
pub extern "C" fn test_packed(a: Packed) {
    hint::black_box(a);
}
