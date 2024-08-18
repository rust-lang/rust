// Test that structs aligned to 128 bits are passed with the correct ABI on powerpc64le.
// This is similar to aarch64-struct-align-128.rs, but for ppc.

//@ compile-flags: --target powerpc64le-unknown-linux-gnu
//@ needs-llvm-components: powerpc

#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_core]

#[lang = "sized"]
trait Sized {}
#[lang = "freeze"]
trait Freeze {}
#[lang = "copy"]
trait Copy {}

#[repr(C)]
pub struct Align8 {
    pub a: u64,
    pub b: u64,
}

#[repr(transparent)]
pub struct Transparent8 {
    a: Align8,
}

#[repr(C)]
pub struct Wrapped8 {
    a: Align8,
}

extern "C" {
    fn test_8(a: Align8, b: Transparent8, c: Wrapped8);
}

#[no_mangle]
fn call_test_8(a: Align8, b: Transparent8, c: Wrapped8) {
    // CHECK: call void @test_8([2 x i64] {{%.*}}, [2 x i64] {{%.*}}, [2 x i64] {{%.*}})
    unsafe { test_8(a, b, c) }
}

#[repr(C)]
#[repr(align(16))]
pub struct Align16 {
    pub a: u64,
    pub b: u64,
}

#[repr(transparent)]
pub struct Transparent16 {
    a: Align16,
}

#[repr(C)]
pub struct Wrapped16 {
    pub a: Align16,
}

extern "C" {
    fn test_16(a: Align16, b: Transparent16, c: Wrapped16);
}

#[no_mangle]
fn call_test_16(a: Align16, b: Transparent16, c: Wrapped16) {
    // It's important that this produces [1 x i128]  rather than just i128!
    // CHECK: call void @test_16([1 x i128] {{%.*}}, [1 x i128] {{%.*}}, [1 x i128] {{%.*}})
    unsafe { test_16(a, b, c) }
}

#[repr(C)]
#[repr(align(32))]
pub struct Align32 {
    pub a: u64,
    pub b: u64,
    pub c: u64,
}

#[repr(transparent)]
pub struct Transparent32 {
    a: Align32,
}

#[repr(C)]
pub struct Wrapped32 {
    pub a: Align32,
}

extern "C" {
    fn test_32(a: Align32, b: Transparent32, c: Wrapped32);
}

#[no_mangle]
fn call_test_32(a: Align32, b: Transparent32, c: Wrapped32) {
    // CHECK: call void @test_32([2 x i128] {{%.*}}, [2 x i128] {{%.*}}, [2 x i128] {{%.*}})
    unsafe { test_32(a, b, c) }
}

pub unsafe fn main(
    a1: Align8,
    a2: Transparent8,
    a3: Wrapped8,
    b1: Align16,
    b2: Transparent16,
    b3: Wrapped16,
    c1: Align32,
    c2: Transparent32,
    c3: Wrapped32,
) {
    call_test_8(a1, a2, a3);
    call_test_16(b1, b2, b3);
    call_test_32(c1, c2, c3);
}
