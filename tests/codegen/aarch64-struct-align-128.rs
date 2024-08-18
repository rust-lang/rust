// Test that structs aligned to 128 bits are passed with the correct ABI on aarch64.

//@ revisions: linux darwin win
//@[linux] compile-flags: --target aarch64-unknown-linux-gnu
//@[darwin] compile-flags: --target aarch64-apple-darwin
//@[win] compile-flags: --target aarch64-pc-windows-msvc
//@[linux] needs-llvm-components: aarch64
//@[darwin] needs-llvm-components: aarch64
//@[win] needs-llvm-components: aarch64

#![feature(no_core, lang_items)]
#![crate_type = "lib"]
#![no_core]

#[lang = "sized"]
trait Sized {}
#[lang = "freeze"]
trait Freeze {}
#[lang = "copy"]
trait Copy {}

// Passed as `[i64 x 2]`, since it's an aggregate with size <= 128 bits, align < 128 bits.
#[repr(C)]
pub struct Align8 {
    pub a: u64,
    pub b: u64,
}

// repr(transparent), so same as above.
#[repr(transparent)]
pub struct Transparent8 {
    a: Align8,
}

// Passed as `[i64 x 2]`, since it's an aggregate with size <= 128 bits, align < 128 bits.
#[repr(C)]
pub struct Wrapped8 {
    a: Align8,
}

extern "C" {
    fn test_8(a: Align8, b: Transparent8, c: Wrapped8);
}

#[no_mangle]
fn call_test_8(a: Align8, b: Transparent8, c: Wrapped8) {
    // linux:  call void @test_8([2 x i64] {{%.*}}, [2 x i64] {{%.*}}, [2 x i64] {{%.*}})
    // darwin: call void @test_8([2 x i64] {{%.*}}, [2 x i64] {{%.*}}, [2 x i64] {{%.*}})
    // win:    call void @test_8([2 x i64] {{%.*}}, [2 x i64] {{%.*}}, [2 x i64] {{%.*}})
    unsafe { test_8(a, b, c) }
}

// Passed as `i128`, since it's an aggregate with size <= 128 bits, align = 128 bits.
// EXCEPT on Linux, where there's a special case to use its unadjusted alignment,
// making it the same as `Align8`, so it's be passed as `[i64 x 2]`.
#[repr(C)]
#[repr(align(16))]
pub struct Align16 {
    pub a: u64,
    pub b: u64,
}

// repr(transparent), so same as above.
#[repr(transparent)]
pub struct Transparent16 {
    a: Align16,
}

// Passed as `i128`, since it's an aggregate with size <= 128 bits, align = 128 bits.
// On Linux, the "unadjustedness" doesn't recurse into fields, so this is passed as `i128`.
#[repr(C)]
pub struct Wrapped16 {
    pub a: Align16,
}

extern "C" {
    fn test_16(a: Align16, b: Transparent16, c: Wrapped16);
}

#[no_mangle]
fn call_test_16(a: Align16, b: Transparent16, c: Wrapped16) {
    // linux:  call void @test_16([2 x i64] {{%.*}}, [2 x i64] {{%.*}}, i128 {{%.*}})
    // darwin: call void @test_16(i128 {{%.*}}, i128 {{%.*}}, i128 {{%.*}})
    // win:    call void @test_16(i128 {{%.*}}, i128 {{%.*}}, i128 {{%.*}})
    unsafe { test_16(a, b, c) }
}

// Passed as `i128`, since it's an aggregate with size <= 128 bits, align = 128 bits.
#[repr(C)]
pub struct I128 {
    pub a: i128,
}

// repr(transparent), so same as above.
#[repr(transparent)]
pub struct TransparentI128 {
    a: I128,
}

// Passed as `i128`, since it's an aggregate with size <= 128 bits, align = 128 bits.
#[repr(C)]
pub struct WrappedI128 {
    pub a: I128,
}

extern "C" {
    fn test_i128(a: I128, b: TransparentI128, c: WrappedI128);
}

#[no_mangle]
fn call_test_i128(a: I128, b: TransparentI128, c: WrappedI128) {
    // linux:  call void @test_i128(i128 {{%.*}}, i128 {{%.*}}, i128 {{%.*}})
    // darwin: call void @test_i128(i128 {{%.*}}, i128 {{%.*}}, i128 {{%.*}})
    // win:    call void @test_i128(i128 {{%.*}}, i128 {{%.*}}, i128 {{%.*}})
    unsafe { test_i128(a, b, c) }
}

// Passed as `[2 x i64]`, since it's an aggregate with size <= 128 bits, align < 128 bits.
// Note that the Linux special case does not apply, because packing is not considered "adjustment".
#[repr(C)]
#[repr(packed)]
pub struct Packed {
    pub a: i128,
}

// repr(transparent), so same as above.
#[repr(transparent)]
pub struct TransparentPacked {
    a: Packed,
}

// Passed as `[2 x i64]`, since it's an aggregate with size <= 128 bits, align < 128 bits.
#[repr(C)]
pub struct WrappedPacked {
    pub a: Packed,
}

extern "C" {
    fn test_packed(a: Packed, b: TransparentPacked, c: WrappedPacked);
}

#[no_mangle]
fn call_test_packed(a: Packed, b: TransparentPacked, c: WrappedPacked) {
    // linux:  call void @test_packed([2 x i64] {{%.*}}, [2 x i64] {{%.*}}, [2 x i64] {{%.*}})
    // darwin: call void @test_packed([2 x i64] {{%.*}}, [2 x i64] {{%.*}}, [2 x i64] {{%.*}})
    // win:    call void @test_packed([2 x i64] {{%.*}}, [2 x i64] {{%.*}}, [2 x i64] {{%.*}})
    unsafe { test_packed(a, b, c) }
}

pub unsafe fn main(
    a1: Align8,
    a2: Transparent8,
    a3: Wrapped8,
    b1: Align16,
    b2: Transparent16,
    b3: Wrapped16,
    c1: I128,
    c2: TransparentI128,
    c3: WrappedI128,
    d1: Packed,
    d2: TransparentPacked,
    d3: WrappedPacked,
) {
    call_test_8(a1, a2, a3);
    call_test_16(b1, b2, b3);
    call_test_i128(c1, c2, c3);
    call_test_packed(d1, d2, d3);
}
