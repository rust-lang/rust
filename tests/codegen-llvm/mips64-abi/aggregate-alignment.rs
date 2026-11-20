//@ add-minicore
//@ compile-flags: -Copt-level=3 -C no-prepopulate-passes
//
//@ revisions: mips64 mips64el
//@[mips64] compile-flags: --target mips64-unknown-linux-gnuabi64
//@[mips64el] compile-flags: --target mips64el-unknown-linux-gnuabi64
//
//@ needs-llvm-components: mips

// Test that 16-byte-aligned aggregates are correctly aligned in registers.

#![feature(no_core)]
#![crate_type = "lib"]
#![expect(unused)]
#![no_core]

extern crate minicore;
use minicore::*;

#[repr(align(16), C)]
struct AlignedStruct<T>(T);

#[repr(align(16), C)]
union AlignedUnion<T> {
    something: ManuallyDrop<T>,
}

type AlignedZSArr = [i128; 0];
type AlignedArr = [i128; 2];

// CHECK: @struct_gets_aligned(i8 %0, ptr [[ZST_PAD_ATTRS:.*]] %_2, i16 %1, i32 %2, [2 x i64] %3)
#[unsafe(no_mangle)]
extern "C" fn struct_gets_aligned(
    arg0: u8,
    arg1: AlignedStruct<()>,
    arg2: u16,
    arg3: AlignedStruct<i8>,
) {
}

// CHECK: @struct_is_aligned(i8 %0, i16 %1, [2 x i64] %2)
#[unsafe(no_mangle)]
extern "C" fn struct_is_aligned(
    arg0: AlignedStruct<()>,
    arg1: u8,
    arg2: i16,
    arg3: AlignedStruct<i8>,
) {
}

// clang 24 gives (i8 noundef signext %arg0, i64 %0)
// CHECK: @trailing_aligned_zst(i8 %0, ptr [[ZST_PAD_ATTRS]] %_2)
#[unsafe(no_mangle)]
extern "C" fn trailing_aligned_zst(arg0: u8, arg1: AlignedStruct<()>) {}

// CHECK: @union_gets_aligned(i8 %0, ptr [[ZST_PAD_ATTRS]] %_2, i16 %1, i32 %2, [2 x i64] %3)
#[unsafe(no_mangle)]
extern "C" fn union_gets_aligned(
    arg0: u8,
    arg1: AlignedUnion<()>,
    arg2: u16,
    arg3: AlignedUnion<i8>,
) {
}

// CHECK: @union_is_aligned(i8 %0, i16 %1, [2 x i64] %2)
#[unsafe(no_mangle)]
extern "C" fn union_is_aligned(
    arg0: AlignedUnion<()>,
    arg1: u8,
    arg2: i16,
    arg3: AlignedUnion<i8>,
) {
}

// CHECK: @array_gets_aligned(i8 %0, ptr [[ZST_PAD_ATTRS]] %_2, i16 %1, i32 %2, [4 x i64] %3)
#[unsafe(no_mangle)]
extern "C" fn array_gets_aligned(arg0: u8, arg1: AlignedZSArr, arg2: u16, arg3: AlignedArr) {}

// CHECK: @array_is_aligned(i8 %0, i16 %1, [4 x i64] %2)
#[unsafe(no_mangle)]
extern "C" fn array_is_aligned(arg0: AlignedZSArr, arg1: u8, arg2: i16, arg3: AlignedArr) {}
