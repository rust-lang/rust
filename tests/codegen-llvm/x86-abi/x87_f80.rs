//@ add-minicore
//
//@ revisions: X86 X86_64 DARWIN WIN64 WIN32
//@ [X86] compile-flags: --target i686-unknown-linux-gnu
//@ [X86] filecheck-flags: --check-prefix ALIGN4
//@ [X86_64] compile-flags: --target x86_64-unknown-linux-gnu
//@ [X86_64] filecheck-flags: --check-prefix ALIGN16
//@ [DARWIN] compile-flags: --target i686-apple-darwin
//@ [DARWIN] filecheck-flags: --check-prefix ALIGN16
//@ [WIN32] compile-flags: --target i686-pc-windows-gnu
//@ [WIN32] filecheck-flags: --check-prefix ALIGN4
//@ [WIN64] compile-flags: --target x86_64-pc-windows-gnu
//@ [WIN64] filecheck-flags: --check-prefix ALIGN16
//@ compile-flags: -Copt-level=3 --crate-type=lib -Zmerge-functions=disabled
//@ needs-llvm-components: x86

#![feature(no_core)]
#![no_std]
#![no_core]
#![allow(non_camel_case_types)]

extern crate minicore;
#[cfg(target_arch = "x86")]
use minicore::arch::x86::{x87_f80, x87_f80_raw};
#[cfg(target_arch = "x86_64")]
use minicore::arch::x86_64::{x87_f80, x87_f80_raw};
use minicore::simd::f64x2;
use minicore::{Copy, MaybeUninit, mem};

// CHECK: %RawStruct = type { [10 x i8], i8 }
#[repr(C)]
pub struct RawStruct(x87_f80_raw, u8);

// ALIGN4: %"minicore::arch::x86::x87_f80" = type { [10 x i8], [2 x i8] }
// ALIGN4: %WrappedStruct = type { %"minicore::arch::x86::x87_f80", i8, [3 x i8] }
//
// ALIGN16: %"minicore::arch::{{x86|x86_64}}::x87_f80" = type { [10 x i8], [6 x i8] }
// ALIGN16: %WrappedStruct = type { %"minicore::arch::{{x86|x86_64}}::x87_f80", i8, [15 x i8] }
#[repr(C)]
pub struct WrappedStruct(x87_f80, u8);

// CHECK: @RAW = external global x86_fp80
// CHECK: @RAW_ARRAY = external global [30 x i8]
// CHECK: @RAW_STRUCT = external global %RawStruct
#[allow(improper_ctypes)]
unsafe extern "C" {
    static RAW: x87_f80_raw;
    static RAW_ARRAY: [x87_f80_raw; 3];
    static RAW_STRUCT: RawStruct;
}

#[unsafe(no_mangle)]
pub fn use_raw_statics() -> (&'static x87_f80_raw, &'static [x87_f80_raw; 3], &'static RawStruct) {
    unsafe { (&RAW, &RAW_ARRAY, &RAW_STRUCT) }
}

// CHECK: @WRAPPED = external global  %"minicore::arch::{{x86|x86_64}}::x87_f80"
// CHECK: @WRAPPED_ARRAY = external global  [3 x %"minicore::arch::{{x86|x86_64}}::x87_f80"]
// CHECK: @WRAPPED_STRUCT = external global %WrappedStruct
#[allow(improper_ctypes)]
unsafe extern "C" {
    static WRAPPED: x87_f80;
    static WRAPPED_ARRAY: [x87_f80; 3];
    static WRAPPED_STRUCT: WrappedStruct;
}

#[unsafe(no_mangle)]
pub fn use_wrapped_statics() -> (&'static x87_f80, &'static [x87_f80; 3], &'static WrappedStruct) {
    unsafe { (&WRAPPED, &WRAPPED_ARRAY, &WRAPPED_STRUCT) }
}

// CHECK-LABEL: size_of_raw_array
#[unsafe(no_mangle)]
pub fn size_of_raw_array() -> usize {
    // CHECK: ret {{i32|i64}} 20
    const { mem::size_of::<[x87_f80_raw; 2]>() }
}

// CHECK-LABEL: raw_array_stride
// CHECK-SAME: dereferenceable(20)
#[unsafe(no_mangle)]
pub fn raw_array_stride(arr: &[x87_f80_raw; 2]) -> x87_f80_raw {
    // CHECK: getelementptr inbounds nuw i8, ptr %arr, {{i32|i64}} 10
    let [_, b] = *arr;
    b
}

#[repr(C)]
struct Single {
    a: x87_f80,
}

#[repr(C)]
struct Pair {
    a: x87_f80,
    b: x87_f80,
}

#[repr(C)]
struct Mixed {
    a: f64,
    b: x87_f80,
}

// X86-LABEL: x86_fp80 @scalar_second(x86_fp80 %{{[0-9]+}}, x86_fp80 returned %{{[0-9]+}})
// X86_64-LABEL: x86_fp80 @scalar_second(x86_fp80 %{{[0-9]+}}, x86_fp80 returned %{{[0-9]+}})
// DARWIN-LABEL: x86_fp80 @scalar_second(x86_fp80 %{{[0-9]+}}, x86_fp80 returned %{{[0-9]+}})
// WIN64-LABEL: void @scalar_second(ptr {{.*}}sret([16 x i8]) {{.*}}, ptr {{.*}}, ptr {{.*}})
// WIN32-LABEL: x86_fp80 @scalar_second(x86_fp80 %{{[0-9]+}}, x86_fp80 returned %{{[0-9]+}})
#[unsafe(no_mangle)]
extern "C" fn scalar_second(_a: x87_f80, b: x87_f80) -> x87_f80 {
    b
}

// X86-LABEL: void @single(ptr {{.*}}sret([12 x i8]) align 4 {{.*}}, ptr {{.*}}byval([12 x i8]) align 4 {{.*}})
// X86_64-LABEL: x86_fp80 @single(ptr {{.*}}byval([16 x i8]) align 16 {{.*}})
// DARWIN-LABEL: void @single(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
// WIN64-LABEL: void @single(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}})
// WIN32-LABEL: void @single(ptr {{.*}}sret([12 x i8]) align 4 {{.*}}, ptr {{.*}}byval([12 x i8]) align 4 {{.*}})
#[unsafe(no_mangle)]
extern "C" fn single(x: Single) -> Single {
    x
}

// X86-LABEL: void @pair(ptr {{.*}}sret([24 x i8]) align 4 {{.*}}, ptr {{.*}}byval([24 x i8]) align 4 {{.*}})
// X86_64-LABEL: void @pair(ptr {{.*}}sret([32 x i8]) align 16 {{.*}}, ptr {{.*}}byval([32 x i8]) align 16 {{.*}})
// DARWIN-LABEL: void @pair(ptr {{.*}}sret([32 x i8]) align 16 {{.*}}, ptr {{.*}}byval([32 x i8]) align 4 {{.*}})
// WIN64-LABEL: void @pair(ptr {{.*}}sret([32 x i8]) align 16 {{.*}}, ptr {{.*}})
// WIN32-LABEL: void @pair(ptr {{.*}}sret([24 x i8]) align 4 {{.*}}, ptr {{.*}}byval([24 x i8]) align 4 {{.*}})
#[unsafe(no_mangle)]
extern "C" fn pair(x: Pair) -> Pair {
    x
}

// Projecting an x87_f80 out of an aggregate used to ICE, because code asserted that
// a field has the same size as the scalar it contains.
//
// CHECK: @pair_first(
#[unsafe(no_mangle)]
fn pair_first(x: Pair) -> x87_f80 {
    x.a
}

// CHECK: @pair_second(
#[unsafe(no_mangle)]
fn pair_second(x: Pair) -> x87_f80 {
    x.b
}

// X86-LABEL: x86_fp80 @mixed(ptr {{.*}}byval([20 x i8]) align 4 {{.*}})
// X86_64-LABEL: x86_fp80 @mixed(ptr {{.*}}byval([32 x i8]) align 16 {{.*}})
// DARWIN-LABEL: x86_fp80 @mixed(ptr {{.*}}byval([32 x i8]) align 4 {{.*}})
// WIN64-LABEL: void @mixed(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}})
// WIN32-LABEL: x86_fp80 @mixed(ptr {{.*}}byval([24 x i8]) align 4 {{.*}})
#[unsafe(no_mangle)]
extern "C" fn mixed(x: Mixed) -> x87_f80 {
    x.b
}

// X86-LABEL: x86_fp80 @many(x86_fp80 %{{[0-9]+}}, x86_fp80 %{{[0-9]+}}, x86_fp80 %{{[0-9]+}}, x86_fp80 %{{[0-9]+}}, x86_fp80 %{{[0-9]+}}, x86_fp80 %{{[0-9]+}}, x86_fp80 %{{[0-9]+}}, x86_fp80 %{{[0-9]+}}, x86_fp80 returned %{{[0-9]+}})
// X86_64-LABEL: x86_fp80 @many(x86_fp80 %{{[0-9]+}}, x86_fp80 %{{[0-9]+}}, x86_fp80 %{{[0-9]+}}, x86_fp80 %{{[0-9]+}}, x86_fp80 %{{[0-9]+}}, x86_fp80 %{{[0-9]+}}, x86_fp80 %{{[0-9]+}}, x86_fp80 %{{[0-9]+}}, x86_fp80 returned %{{[0-9]+}})
// DARWIN-LABEL: x86_fp80 @many(x86_fp80 %{{[0-9]+}}, x86_fp80 %{{[0-9]+}}, x86_fp80 %{{[0-9]+}}, x86_fp80 %{{[0-9]+}}, x86_fp80 %{{[0-9]+}}, x86_fp80 %{{[0-9]+}}, x86_fp80 %{{[0-9]+}}, x86_fp80 %{{[0-9]+}}, x86_fp80 returned %{{[0-9]+}})
// WIN64-LABEL: void @many(ptr {{.*}}sret([16 x i8]) {{.*}}
// WIN32-LABEL: x86_fp80 @many(x86_fp80 %{{[0-9]+}}, x86_fp80 %{{[0-9]+}}, x86_fp80 %{{[0-9]+}}, x86_fp80 %{{[0-9]+}}, x86_fp80 %{{[0-9]+}}, x86_fp80 %{{[0-9]+}}, x86_fp80 %{{[0-9]+}}, x86_fp80 %{{[0-9]+}}, x86_fp80 returned %{{[0-9]+}})
#[unsafe(no_mangle)]
extern "C" fn many(
    _a: x87_f80,
    _b: x87_f80,
    _c: x87_f80,
    _d: x87_f80,
    _e: x87_f80,
    _f: x87_f80,
    _g: x87_f80,
    _h: x87_f80,
    i: x87_f80,
) -> x87_f80 {
    i
}

/// Unions test the overlapping of X87 with SSE/INTEGER register classes.
///
/// The ABI logic processes the fields in lexicographical order, so check
/// both the x87_f80 coming first and coming last.
#[repr(C)]
union X87Before<T: Copy> {
    a: x87_f80,
    b: T,
}

#[repr(C)]
union X87After<T: Copy> {
    b: T,
    a: x87_f80,
}

// X86-LABEL: void @union_x87_with_f64(ptr {{.*}}sret([12 x i8]) align 4 {{.*}}, ptr {{.*}}byval([12 x i8]) align 4 {{.*}})
// X86_64-LABEL: void @union_x87_with_f64(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 16 {{.*}})
// DARWIN-LABEL: void @union_x87_with_f64(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
// WIN64-LABEL: void @union_x87_with_f64(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}})
// WIN32-LABEL: void @union_x87_with_f64(ptr {{.*}}sret([16 x i8]) align 8 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
#[unsafe(no_mangle)]
extern "C" fn union_x87_with_f64(x: X87Before<f64>) -> X87Before<f64> {
    x
}

// X86-LABEL: void @union_x87_with_u64(ptr {{.*}}sret([12 x i8]) align 4 {{.*}}, ptr {{.*}}byval([12 x i8]) align 4 {{.*}})
// X86_64-LABEL: void @union_x87_with_u64(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 16 {{.*}})
// DARWIN-LABEL: void @union_x87_with_u64(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
// WIN64-LABEL: void @union_x87_with_u64(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}})
// WIN32-LABEL: void @union_x87_with_u64(ptr {{.*}}sret([16 x i8]) align 8 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
#[unsafe(no_mangle)]
extern "C" fn union_x87_with_u64(x: X87Before<u64>) -> X87Before<u64> {
    x
}

#[repr(C)]
struct F64F64(f64, f64);
impl Copy for F64F64 {}

// X86-LABEL: void @x87_before_f64_f64(ptr {{.*}}sret([16 x i8]) align 4 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
// X86_64-LABEL: void @x87_before_f64_f64(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 16 {{.*}})
// DARWIN-LABEL: void @x87_before_f64_f64(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
// WIN64-LABEL: void @x87_before_f64_f64(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}})
// WIN32-LABEL: void @x87_before_f64_f64(ptr {{.*}}sret([16 x i8]) align 8 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
#[unsafe(no_mangle)]
extern "C" fn x87_before_f64_f64(x: X87Before<F64F64>) -> X87Before<F64F64> {
    x
}

// X86-LABEL: void @x87_after_f64_f64(ptr {{.*}}sret([16 x i8]) align 4 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
// X86_64-LABEL: void @x87_after_f64_f64(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 16 {{.*}})
// DARWIN-LABEL: void @x87_after_f64_f64(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
// WIN64-LABEL: void @x87_after_f64_f64(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}})
// WIN32-LABEL: void @x87_after_f64_f64(ptr {{.*}}sret([16 x i8]) align 8 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
#[unsafe(no_mangle)]
extern "C" fn x87_after_f64_f64(x: X87After<F64F64>) -> X87After<F64F64> {
    x
}

#[repr(C)]
struct U64F64(u64, f64);
impl Copy for U64F64 {}

// X86-LABEL: void @x87_before_u64_f64(ptr {{.*}}sret([16 x i8]) align 4 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
// X86_64-LABEL: void @x87_before_u64_f64(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 16 {{.*}})
// DARWIN-LABEL: void @x87_before_u64_f64(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
// WIN64-LABEL: void @x87_before_u64_f64(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}})
// WIN32-LABEL: void @x87_before_u64_f64(ptr {{.*}}sret([16 x i8]) align 8 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
#[unsafe(no_mangle)]
extern "C" fn x87_before_u64_f64(x: X87Before<U64F64>) -> X87Before<U64F64> {
    x
}

// X86-LABEL: void @x87_after_u64_f64(ptr {{.*}}sret([16 x i8]) align 4 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
// X86_64-LABEL: void @x87_after_u64_f64(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 16 {{.*}})
// DARWIN-LABEL: void @x87_after_u64_f64(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
// WIN64-LABEL: void @x87_after_u64_f64(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}})
// WIN32-LABEL: void @x87_after_u64_f64(ptr {{.*}}sret([16 x i8]) align 8 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
#[unsafe(no_mangle)]
extern "C" fn x87_after_u64_f64(x: X87After<U64F64>) -> X87After<U64F64> {
    x
}

#[repr(C)]
struct U64U64(u64, u64);
impl Copy for U64U64 {}

// X86-LABEL: void @x87_before_u64_u64(ptr {{.*}}sret([16 x i8]) align 4 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
// X86_64-LABEL: { i64, i64 } @x87_before_u64_u64({ i64, i64 } {{.*}})
// DARWIN-LABEL: void @x87_before_u64_u64(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
// WIN64-LABEL: void @x87_before_u64_u64(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}})
// WIN32-LABEL: void @x87_before_u64_u64(ptr {{.*}}sret([16 x i8]) align 8 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
#[unsafe(no_mangle)]
extern "C" fn x87_before_u64_u64(x: X87Before<U64U64>) -> X87Before<U64U64> {
    x
}

// X86-LABEL: void @x87_before_f64x2(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
// X86_64-LABEL: void @x87_before_f64x2(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 16 {{.*}})
// DARWIN-LABEL: void @x87_before_f64x2(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 16 {{.*}})
// WIN64-LABEL: void @x87_before_f64x2(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}})
// WIN32-LABEL: void @x87_before_f64x2(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
#[unsafe(no_mangle)]
extern "C" fn x87_before_f64x2(x: X87Before<f64x2>) -> X87Before<f64x2> {
    x
}

// X86-LABEL: void @x87_after_f64x2(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
// X86_64-LABEL: void @x87_after_f64x2(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 16 {{.*}})
// DARWIN-LABEL: void @x87_after_f64x2(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 16 {{.*}})
// WIN64-LABEL: void @x87_after_f64x2(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}})
// WIN32-LABEL: void @x87_after_f64x2(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
#[unsafe(no_mangle)]
extern "C" fn x87_after_f64x2(x: X87After<f64x2>) -> X87After<f64x2> {
    x
}

// X86-LABEL: x86_fp80 @array_third(
// X86: getelementptr inbounds{{( nuw)?}} i8, ptr %xs, i32 24
// X86-NEXT: load x86_fp80, ptr {{.*}}, align 4
//
// X86_64-LABEL: x86_fp80 @array_third(
// X86_64: getelementptr inbounds{{( nuw)?}} i8, ptr %xs, i64 32
// X86_64-NEXT: load x86_fp80, ptr {{.*}}, align 16
//
// DARWIN-LABEL: x86_fp80 @array_third(
// DARWIN: getelementptr inbounds{{( nuw)?}} i8, ptr %xs, i32 32
// DARWIN-NEXT: load x86_fp80, ptr {{.*}}, align 16
//
// WIN64-LABEL: void @array_third(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}align 16 {{.*}}%xs)
// WIN64: getelementptr inbounds{{( nuw)?}} i8, ptr %xs, i64 32
// WIN64-NEXT: call void @llvm.memcpy{{.*}}(ptr {{.*}}, ptr {{.*}}, i64 16, i1 false)
//
// WIN32-LABEL: x86_fp80 @array_third(
// WIN32: getelementptr inbounds{{( nuw)?}} i8, ptr %xs, i32 24
// WIN32-NEXT: load x86_fp80, ptr {{.*}}, align 4
#[unsafe(no_mangle)]
extern "C" fn array_third(xs: &[x87_f80; 4]) -> x87_f80 {
    let [_, _, c, _] = *xs;
    c
}

// CHECK-LABEL: @array_copy(
// X86: call void @llvm.memcpy{{.*}}(ptr {{.*}}, ptr {{.*}}, i32 48, i1 false)
// X86_64: call void @llvm.memcpy{{.*}}(ptr {{.*}}, ptr {{.*}}, i64 64, i1 false)
// DARWIN: call void @llvm.memcpy{{.*}}(ptr {{.*}}, ptr {{.*}}, i32 64, i1 false)
// WIN64: call void @llvm.memcpy{{.*}}(ptr {{.*}}, ptr {{.*}}, i64 64, i1 false)
// WIN32: call void @llvm.memcpy{{.*}}(ptr {{.*}}, ptr {{.*}}, i32 48, i1 false)
#[unsafe(no_mangle)]
fn array_copy(xs: &[x87_f80; 4], ys: &mut [x87_f80; 4]) {
    *ys = *xs;
}

#[repr(transparent)]
struct TransparentWrapper<T>(T);

// X86-LABEL: x86_fp80 @transparent_struct(x86_fp80 returned %{{[0-9]+}})
// X86_64-LABEL: x86_fp80 @transparent_struct(x86_fp80 returned %{{[0-9]+}})
// DARWIN-LABEL: x86_fp80 @transparent_struct(x86_fp80 returned %{{[0-9]+}})
// WIN64-LABEL: void @transparent_struct(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}align 16 {{.*}})
// WIN32-LABEL: x86_fp80 @transparent_struct(x86_fp80 returned %{{[0-9]+}})
#[unsafe(no_mangle)]
extern "C" fn transparent_struct(x: TransparentWrapper<x87_f80>) -> TransparentWrapper<x87_f80> {
    x
}

// X86-LABEL: x86_fp80 @transparent_union(x86_fp80 returned %{{[0-9]+}})
// X86_64-LABEL: x86_fp80 @transparent_union(x86_fp80 returned %{{[0-9]+}})
// DARWIN-LABEL: x86_fp80 @transparent_union(x86_fp80 returned %{{[0-9]+}})
// WIN64-LABEL: void @transparent_union(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}align 16 {{.*}})
// WIN32-LABEL: x86_fp80 @transparent_union(x86_fp80 returned %{{[0-9]+}})
#[unsafe(no_mangle)]
extern "C" fn transparent_union(x: MaybeUninit<x87_f80>) -> MaybeUninit<x87_f80> {
    x
}
