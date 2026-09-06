// Checks that small aggregates returned as an integer under the Win32 (MSVC) ABI
// (`abi_return_struct_as_int`) carry `noundef` when the layout is provably free of
// uninit bytes, and omit it when the layout may contain uninit bytes (padding or a
// union).
//
// See <https://github.com/rust-lang/rust/issues/123183>.

//@ add-minicore
//@ compile-flags: --target i686-pc-windows-msvc -Cno-prepopulate-passes -Copt-level=3
//@ needs-llvm-components: x86

#![feature(no_core, lang_items)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

// Fully-defined layouts, one per integer register size (i8/i16/i32/i64).

#[repr(C)]
pub struct OneU8 {
    x: u8,
}

#[repr(C)]
pub struct TwoU8 {
    x: u8,
    y: u8,
}

#[repr(C)]
pub struct TwoU16 {
    x: u16,
    y: u16,
}

#[repr(C)]
pub struct TwoU32 {
    x: u32,
    y: u32,
}

// Layouts that may contain uninit bytes.

// Tail padding: `u8` at offset 4 leaves bytes 5..8 uninit.
#[repr(C)]
pub struct Padded {
    x: u32,
    y: u8,
}

#[repr(C)]
pub union U {
    x: u32,
    y: u32,
}

// CHECK: define noundef i8 @ret_one_u8()
#[no_mangle]
pub extern "C" fn ret_one_u8() -> OneU8 {
    OneU8 { x: 1 }
}

// CHECK: define noundef i16 @ret_two_u8()
#[no_mangle]
pub extern "C" fn ret_two_u8() -> TwoU8 {
    TwoU8 { x: 1, y: 2 }
}

// CHECK: define noundef i32 @ret_two_u16()
#[no_mangle]
pub extern "C" fn ret_two_u16() -> TwoU16 {
    TwoU16 { x: 1, y: 2 }
}

// CHECK: define noundef i64 @ret_two_u32()
#[no_mangle]
pub extern "C" fn ret_two_u32() -> TwoU32 {
    TwoU32 { x: 1, y: 2 }
}

// Tail padding -> no `noundef`.
// CHECK: define i64 @ret_padded()
#[no_mangle]
pub extern "C" fn ret_padded() -> Padded {
    Padded { x: 1, y: 2 }
}

// Union -> no `noundef`.
// CHECK: define i32 @ret_union()
#[no_mangle]
pub extern "C" fn ret_union() -> U {
    U { x: 1 }
}
