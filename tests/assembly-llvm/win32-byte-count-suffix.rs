//@ add-minicore
//@ assembly-output: emit-asm
//@ compile-flags: -Z merge-functions=disabled

//@ revisions: windows-gnu windows-msvc
//@[windows-gnu] compile-flags: --target i686-pc-windows-gnu
//@[windows-gnu] needs-llvm-components: x86
//
//@[windows-msvc] compile-flags: --target i686-pc-windows-msvc
//@[windows-msvc] needs-llvm-components: x86

#![feature(no_core, rustc_attrs, abi_vectorcall)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::simd::i32x4;
use minicore::*;

// CHECK-LABEL: _basic:
#[unsafe(no_mangle)]
extern "C" fn basic(_: u64) {}

// CHECK-LABEL: _basic_stdcall@8:
#[unsafe(no_mangle)]
extern "stdcall" fn basic_stdcall(_: u64) {}

// CHECK-LABEL: _basic_system@8:
#[unsafe(no_mangle)]
extern "system" fn basic_system(_: u64) {}

// CHECK-LABEL: _basic_thiscall:
#[unsafe(no_mangle)]
extern "thiscall" fn basic_thiscall(_: u64) {}

// CHECK-LABEL: @basic_fastcall@8:
#[unsafe(no_mangle)]
extern "fastcall" fn basic_fastcall(_: u64) {}

// CHECK-LABEL: basic_vectorcall@@8:
#[unsafe(no_mangle)]
extern "vectorcall" fn basic_vectorcall(_: u64) {}

// CHECK-LABEL: raw:
#[unsafe(export_name = "\x01raw")]
extern "C" fn raw(_: u64) {}

// CHECK-LABEL: raw_stdcall:
#[unsafe(export_name = "\x01raw_stdcall")]
extern "stdcall" fn raw_stdcall(_: u64) {}

// CHECK-LABEL: raw_fastcall:
#[unsafe(export_name = "\x01raw_fastcall")]
extern "fastcall" fn raw_fastcall(_: u64) {}

// CHECK-LABEL: raw_vectorcall:
#[unsafe(export_name = "\x01raw_vectorcall")]
extern "vectorcall" fn raw_vectorcall(_: u64) {}

// CHECK-LABEL: "?question_mark":
#[unsafe(export_name = "?question_mark")]
extern "C" fn question_mark(_: u64) {}

// CHECK-LABEL: "?question_mark_stdcall":
#[unsafe(export_name = "?question_mark_stdcall")]
extern "stdcall" fn question_mark_stdcall(_: u64) {}

// CHECK-LABEL: "?question_mark_fastcall":
#[unsafe(export_name = "?question_mark_fastcall")]
extern "fastcall" fn question_mark_fastcall(_: u64) {}

// CHECK-LABEL: "?question_mark_vectorcall":
#[unsafe(export_name = "?question_mark_vectorcall")]
extern "vectorcall" fn question_mark_vectorcall(_: u64) {}

// CHECK-LABEL: _bytes:
#[unsafe(no_mangle)]
extern "C" fn bytes(_: u8, _: u8, _: u8) {}

// CHECK-LABEL: _bytes_stdcall@12:
#[unsafe(no_mangle)]
extern "stdcall" fn bytes_stdcall(_: u8, _: u8, _: u8) {}

// CHECK-LABEL: @bytes_fastcall@12:
#[unsafe(no_mangle)]
extern "fastcall" fn bytes_fastcall(_: u8, _: u8, _: u8) {}

// CHECK-LABEL: bytes_vectorcall@@12:
#[unsafe(no_mangle)]
extern "vectorcall" fn bytes_vectorcall(_: u8, _: u8, _: u8) {}

// CHECK-LABEL: _int128:
#[unsafe(no_mangle)]
extern "C" fn int128(_: u128) {}

// CHECK-LABEL: _int128_stdcall@16:
#[unsafe(no_mangle)]
extern "stdcall" fn int128_stdcall(_: u128) {}

// CHECK-LABEL: @int128_fastcall@16:
#[unsafe(no_mangle)]
extern "fastcall" fn int128_fastcall(_: u128) {}

// CHECK-LABEL: int128_vectorcall@@16:
#[unsafe(no_mangle)]
extern "vectorcall" fn int128_vectorcall(_: u128) {}

// CHECK-LABEL: _no_arguments:
#[unsafe(no_mangle)]
extern "C" fn no_arguments() {}

// CHECK-LABEL: _no_arguments_stdcall@0:
#[unsafe(no_mangle)]
extern "stdcall" fn no_arguments_stdcall() {}

// CHECK-LABEL: @no_arguments_fastcall@0:
#[unsafe(no_mangle)]
extern "fastcall" fn no_arguments_fastcall() {}

// CHECK-LABEL: no_arguments_vectorcall@@0:
#[unsafe(no_mangle)]
extern "vectorcall" fn no_arguments_vectorcall() {}

// CHECK-LABEL: _rounding:
#[unsafe(no_mangle)]
extern "C" fn rounding(_: u8, _: f64) {}

// CHECK-LABEL: _rounding_stdcall@12:
#[unsafe(no_mangle)]
extern "stdcall" fn rounding_stdcall(_: u8, _: f64) {}

// CHECK-LABEL: @rounding_fastcall@12:
#[unsafe(no_mangle)]
extern "fastcall" fn rounding_fastcall(_: u8, _: f64) {}

// CHECK-LABEL: rounding_vectorcall@@12:
#[unsafe(no_mangle)]
extern "vectorcall" fn rounding_vectorcall(_: u8, _: f64) {}

// NOTE: even with naked functions we can't currently define c-variadic
// variants for the other calling conventions.
//
// CHECK-LABEL: _variadic:
#[unsafe(no_mangle)]
unsafe extern "C" fn variadic(x: u64, _: ...) -> u64 {
    x
}

#[repr(C)]
struct S {
    a: u64,
    b: u64,
    c: u64,
}

// CHECK-LABEL: _sret:
#[unsafe(no_mangle)]
extern "C" fn sret(x: u64) -> S {
    S { a: x, b: x, c: x }
}

// CHECK-LABEL: _sret_stdcall@8:
#[unsafe(no_mangle)]
extern "stdcall" fn sret_stdcall(x: u64) -> S {
    S { a: x, b: x, c: x }
}

// CHECK-LABEL: @sret_fastcall@8:
#[unsafe(no_mangle)]
extern "fastcall" fn sret_fastcall(x: u64) -> S {
    S { a: x, b: x, c: x }
}

// CHECK-LABEL: sret_vectorcall@@8:
#[unsafe(no_mangle)]
extern "vectorcall" fn sret_vectorcall(x: u64) -> S {
    S { a: x, b: x, c: x }
}

#[repr(C)]
struct ZST;

// CHECK-LABEL: _zst_struct:
#[unsafe(no_mangle)]
extern "C" fn zst_struct(_: ZST) {}

// CHECK-LABEL: _zst_struct_stdcall@0:
#[unsafe(no_mangle)]
extern "stdcall" fn zst_struct_stdcall(_: ZST) {}

// CHECK-LABEL: @zst_struct_fastcall@0:
#[unsafe(no_mangle)]
extern "fastcall" fn zst_struct_fastcall(_: ZST) {}

// CHECK-LABEL: zst_struct_vectorcall@@0:
#[unsafe(no_mangle)]
extern "vectorcall" fn zst_struct_vectorcall(_: ZST) {}

// CHECK-LABEL: _zst_array:
#[unsafe(no_mangle)]
extern "C" fn zst_array(_: [u8; 0]) {}

// CHECK-LABEL: _zst_array_stdcall@0:
#[unsafe(no_mangle)]
extern "stdcall" fn zst_array_stdcall(_: [u8; 0]) {}

// CHECK-LABEL: @zst_array_fastcall@0:
#[unsafe(no_mangle)]
extern "fastcall" fn zst_array_fastcall(_: [u8; 0]) {}

// CHECK-LABEL: zst_array_vectorcall@@0:
#[unsafe(no_mangle)]
extern "vectorcall" fn zst_array_vectorcall(_: [u8; 0]) {}

// CHECK-LABEL: _four_vectors:
#[unsafe(no_mangle)]
#[target_feature(enable = "sse2")]
extern "C" fn four_vectors(_: i32x4, _: i32x4, _: i32x4, _: i32x4) {}

/// FIXME: on msvc we should pass the 4th argument indirectly, but currently don't.
/// When we do, the total count should remain 64.
// CHECK-LABEL: _four_vectors_stdcall@64:
#[unsafe(no_mangle)]
#[target_feature(enable = "sse2")]
extern "stdcall" fn four_vectors_stdcall(_: i32x4, _: i32x4, _: i32x4, _: i32x4) {}

// CHECK-LABEL: @four_vectors_fastcall@64:
#[unsafe(no_mangle)]
#[target_feature(enable = "sse2")]
extern "fastcall" fn four_vectors_fastcall(_: i32x4, _: i32x4, _: i32x4, _: i32x4) {}

// CHECK-LABEL: four_vectors_vectorcall@@64:
#[unsafe(no_mangle)]
#[target_feature(enable = "sse2")]
extern "vectorcall" fn four_vectors_vectorcall(_: i32x4, _: i32x4, _: i32x4, _: i32x4) {}

// CHECK-LABEL: eight_vectors_vectorcall@@128:
#[unsafe(no_mangle)]
#[target_feature(enable = "sse2")]
extern "vectorcall" fn eight_vectors_vectorcall(
    _: i32x4,
    _: i32x4,
    _: i32x4,
    _: i32x4,
    _: i32x4,
    _: i32x4,
    _: i32x4,
    _: i32x4,
) {
}

/// FIXME: the mangled name for this type is incorrect on msvc, where the argument is passed
/// indirectly and a pointer takes up 4 bytes, but clang counts the size of the type and uses
/// @8 there. With GNU the value is passed directly, so the suffix is correct.
#[repr(C, align(8))]
struct OverAligned(u32, u32);

// CHECK-LABEL: _overaligned:
#[unsafe(no_mangle)]
extern "C" fn overaligned(_: OverAligned) {}

// windows-gnu-LABEL: _overaligned_stdcall@8:
// windows-msvc-LABEL: _overaligned_stdcall@4:
#[unsafe(no_mangle)]
extern "stdcall" fn overaligned_stdcall(_: OverAligned) {}

// windows-gnu-LABEL: @overaligned_fastcall@8:
// windows-msvc-LABEL: @overaligned_fastcall@4:
#[unsafe(no_mangle)]
extern "fastcall" fn overaligned_fastcall(_: OverAligned) {}

// windows-gnu-LABEL: overaligned_vectorcall@@8:
// windows-msvc-LABEL: overaligned_vectorcall@@4:
#[unsafe(no_mangle)]
extern "vectorcall" fn overaligned_vectorcall(_: OverAligned) {}
