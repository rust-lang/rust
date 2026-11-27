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
struct AlignedStruct16<T>(T);

#[repr(align(8), C)]
struct AlignedStruct8<T>(T);

#[repr(align(16), C)]
union AlignedUnion16<T> {
    something: ManuallyDrop<T>,
}

#[repr(align(8), C)]
union AlignedUnion8<T> {
    something: ManuallyDrop<T>,
}

type AlignedArr = [i128; 2];
type Arr = [i64; 2];

// CHECK: @struct_gets_aligned(i8 %0, ptr [[ZST_PAD_ATTRS:.*]] %_2, i16 %1, i32 %2, [2 x i64] %3)
#[unsafe(no_mangle)]
extern "C" fn struct_gets_aligned(
    arg0: u8,
    arg1: AlignedStruct16<()>,
    arg2: u16,
    arg3: AlignedStruct16<i8>,
) {
}

// CHECK: @struct_gets_aligned_before_varargs(i8 %0, ptr [[ZST_PAD_ATTRS]] %_2, ...)
#[unsafe(no_mangle)]
unsafe extern "C" fn struct_gets_aligned_before_varargs(
    arg0: u8,
    arg1: AlignedStruct16<()>,
    arg2: ...
) {
}

// CHECK: @struct_trivially_aligned(i8 %0, i16 %1, i64 %2)
#[unsafe(no_mangle)]
extern "C" fn struct_trivially_aligned(
    arg0: u8,
    arg1: AlignedStruct8<()>,
    arg2: u16,
    arg3: AlignedStruct8<i8>,
) {
}

// CHECK: @struct_trivially_aligned_before_varargs(i8 %0, ...)
#[unsafe(no_mangle)]
unsafe extern "C" fn struct_trivially_aligned_before_varargs(
    arg0: u8,
    arg1: AlignedStruct8<()>,
    arg2: ...
) {
}

// CHECK: @struct_is_aligned(i8 %0, i16 %1, [2 x i64] %2)
#[unsafe(no_mangle)]
extern "C" fn struct_is_aligned(
    arg0: AlignedStruct16<()>,
    arg1: u8,
    arg2: i16,
    arg3: AlignedStruct16<i8>,
) {
}

// CHECK: @struct_is_aligned_before_varargs(i8 %0, ...)
#[unsafe(no_mangle)]
unsafe extern "C" fn struct_is_aligned_before_varargs(
    arg0: AlignedStruct16<()>,
    arg1: u8,
    arg2: ...
) {
}

// clang 24 gives (i8 noundef signext %arg0, i64 %0)
// CHECK: @trailing_aligned_zst(i8 %0, ptr [[ZST_PAD_ATTRS]] %_2)
#[unsafe(no_mangle)]
extern "C" fn trailing_aligned_zst(arg0: u8, arg1: AlignedStruct16<()>) {}

// CHECK: @union_gets_aligned(i8 %0, ptr [[ZST_PAD_ATTRS]] %_2, i16 %1, i32 %2, [2 x i64] %3)
#[unsafe(no_mangle)]
extern "C" fn union_gets_aligned(
    arg0: u8,
    arg1: AlignedUnion16<()>,
    arg2: u16,
    arg3: AlignedUnion16<i8>,
) {
}

// CHECK: @union_gets_aligned_before_varargs(i8 %0, ptr [[ZST_PAD_ATTRS]] %_2, ...)
#[unsafe(no_mangle)]
unsafe extern "C" fn union_gets_aligned_before_varargs(
    arg0: u8,
    arg1: AlignedUnion16<()>,
    arg2: ...
) {
}

// CHECK: @union_trivially_aligned(i8 %0, i16 %1, i64 %2)
#[unsafe(no_mangle)]
extern "C" fn union_trivially_aligned(
    arg0: u8,
    arg1: AlignedUnion8<()>,
    arg2: u16,
    arg3: AlignedUnion8<i8>,
) {
}

// CHECK: @union_trivially_aligned_before_varargs(i8 %0, ...)
#[unsafe(no_mangle)]
unsafe extern "C" fn union_trivially_aligned_before_varargs(
    arg0: u8,
    arg1: AlignedUnion8<()>,
    arg2: ...
) {
}

// CHECK: @union_is_aligned(i8 %0, i16 %1, [2 x i64] %2)
#[unsafe(no_mangle)]
extern "C" fn union_is_aligned(
    arg0: AlignedUnion16<()>,
    arg1: u8,
    arg2: i16,
    arg3: AlignedUnion16<i8>,
) {
}

// CHECK: @union_is_aligned_before_varargs(i8 %0, ...)
#[unsafe(no_mangle)]
unsafe extern "C" fn union_is_aligned_before_varargs(
    arg0: AlignedUnion16<()>,
    arg1: u8,
    arg2: ...
) {
}

// CHECK: @array_gets_aligned(i8 %0, i32 %1, [4 x i64] %2)
#[unsafe(no_mangle)]
extern "C" fn array_gets_aligned(arg0: u8, arg1: AlignedArr) {}

// CHECK: @array_trivially_aligned(i8 %0, [2 x i64] %1)
#[unsafe(no_mangle)]
extern "C" fn array_trivially_aligned(arg0: u8, arg1: Arr) {}

// CHECK: @array_is_aligned([4 x i64] %0, i8 %1)
#[unsafe(no_mangle)]
extern "C" fn array_is_aligned(arg0: AlignedArr, arg1: u8) {}
