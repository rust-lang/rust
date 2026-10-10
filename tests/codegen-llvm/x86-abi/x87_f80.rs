//@ add-minicore
//
//@ revisions: X86 X86_64 DARWIN WIN64
//@ [X86] compile-flags: --target i686-unknown-linux-gnu
//@ [X86_64] compile-flags: --target x86_64-unknown-linux-gnu
//@ [DARWIN] compile-flags: --target i686-apple-darwin
//@ [WIN64] compile-flags: --target x86_64-pc-windows-gnu
//@ compile-flags: -Copt-level=3 --crate-type=lib -Zmerge-functions=disabled
//@ needs-llvm-components: x86

#![feature(no_core)]
#![no_std]
#![no_core]
#![allow(non_camel_case_types)]

extern crate minicore;
use minicore::Copy;
#[cfg(target_arch = "x86")]
use minicore::arch::x86::x87_f80;
#[cfg(target_arch = "x86_64")]
use minicore::arch::x86_64::x87_f80;
use minicore::simd::f64x2;

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

// X86-LABEL: x86_fp80 @scalar_second(x86_fp80 noundef %_a, x86_fp80 noundef returned %b)
// X86_64-LABEL: x86_fp80 @scalar_second(x86_fp80 noundef %_a, x86_fp80 noundef returned %b)
// DARWIN-LABEL: x86_fp80 @scalar_second(x86_fp80 noundef %_a, x86_fp80 noundef returned %b)
// WIN64-LABEL: void @scalar_second(ptr {{.*}}sret([16 x i8]) {{.*}}, ptr {{.*}}, ptr {{.*}})
#[unsafe(no_mangle)]
extern "C" fn scalar_second(_a: x87_f80, b: x87_f80) -> x87_f80 {
    b
}

// X86-LABEL: void @single(ptr {{.*}}sret([12 x i8]) align 4 {{.*}}, ptr {{.*}}byval([12 x i8]) align 4 {{.*}})
// X86_64-LABEL: x86_fp80 @single(ptr {{.*}}byval([16 x i8]) align 16 {{.*}})
// DARWIN-LABEL: void @single(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
// WIN64-LABEL: void @single(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}})
#[unsafe(no_mangle)]
extern "C" fn single(x: Single) -> Single {
    x
}

// X86-LABEL: void @pair(ptr {{.*}}sret([24 x i8]) align 4 {{.*}}, ptr {{.*}}byval([24 x i8]) align 4 {{.*}})
// X86_64-LABEL: void @pair(ptr {{.*}}sret([32 x i8]) align 16 {{.*}}, ptr {{.*}}byval([32 x i8]) align 16 {{.*}})
// DARWIN-LABEL: void @pair(ptr {{.*}}sret([32 x i8]) align 16 {{.*}}, ptr {{.*}}byval([32 x i8]) align 4 {{.*}})
// WIN64-LABEL: void @pair(ptr {{.*}}sret([32 x i8]) align 16 {{.*}}, ptr {{.*}})
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
#[unsafe(no_mangle)]
extern "C" fn mixed(x: Mixed) -> x87_f80 {
    x.b
}

// X86-LABEL: x86_fp80 @many(x86_fp80 noundef %_a, x86_fp80 noundef %_b, x86_fp80 noundef %_c, x86_fp80 noundef %_d, x86_fp80 noundef %_e, x86_fp80 noundef %_f, x86_fp80 noundef %_g, x86_fp80 noundef %_h, x86_fp80 noundef returned %i)
// X86_64-LABEL: x86_fp80 @many(x86_fp80 noundef %_a, x86_fp80 noundef %_b, x86_fp80 noundef %_c, x86_fp80 noundef %_d, x86_fp80 noundef %_e, x86_fp80 noundef %_f, x86_fp80 noundef %_g, x86_fp80 noundef %_h, x86_fp80 noundef returned %i)
// DARWIN-LABEL: x86_fp80 @many(x86_fp80 noundef %_a, x86_fp80 noundef %_b, x86_fp80 noundef %_c, x86_fp80 noundef %_d, x86_fp80 noundef %_e, x86_fp80 noundef %_f, x86_fp80 noundef %_g, x86_fp80 noundef %_h, x86_fp80 noundef returned %i)
// WIN64-LABEL: void @many(ptr {{.*}}sret([16 x i8]) {{.*}}
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
#[unsafe(no_mangle)]
extern "C" fn union_x87_with_f64(x: X87Before<f64>) -> X87Before<f64> {
    x
}

// X86-LABEL: void @union_x87_with_u64(ptr {{.*}}sret([12 x i8]) align 4 {{.*}}, ptr {{.*}}byval([12 x i8]) align 4 {{.*}})
// X86_64-LABEL: void @union_x87_with_u64(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 16 {{.*}})
// DARWIN-LABEL: void @union_x87_with_u64(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
// WIN64-LABEL: void @union_x87_with_u64(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}})
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
#[unsafe(no_mangle)]
extern "C" fn x87_before_f64_f64(x: X87Before<F64F64>) -> X87Before<F64F64> {
    x
}

// X86-LABEL: void @x87_after_f64_f64(ptr {{.*}}sret([16 x i8]) align 4 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
// X86_64-LABEL: void @x87_after_f64_f64(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 16 {{.*}})
// DARWIN-LABEL: void @x87_after_f64_f64(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
// WIN64-LABEL: void @x87_after_f64_f64(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}})
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
#[unsafe(no_mangle)]
extern "C" fn x87_before_u64_f64(x: X87Before<U64F64>) -> X87Before<U64F64> {
    x
}

// X86-LABEL: void @x87_after_u64_f64(ptr {{.*}}sret([16 x i8]) align 4 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
// X86_64-LABEL: void @x87_after_u64_f64(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 16 {{.*}})
// DARWIN-LABEL: void @x87_after_u64_f64(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
// WIN64-LABEL: void @x87_after_u64_f64(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}})
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
#[unsafe(no_mangle)]
extern "C" fn x87_before_u64_u64(x: X87Before<U64U64>) -> X87Before<U64U64> {
    x
}

// X86-LABEL: void @x87_before_f64x2(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
// X86_64-LABEL: void @x87_before_f64x2(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 16 {{.*}})
// DARWIN-LABEL: void @x87_before_f64x2(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 16 {{.*}})
// WIN64-LABEL: void @x87_before_f64x2(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}})
#[unsafe(no_mangle)]
extern "C" fn x87_before_f64x2(x: X87Before<f64x2>) -> X87Before<f64x2> {
    x
}

// X86-LABEL: void @x87_after_f64x2(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 4 {{.*}})
// X86_64-LABEL: void @x87_after_f64x2(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 16 {{.*}})
// DARWIN-LABEL: void @x87_after_f64x2(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}}byval([16 x i8]) align 16 {{.*}})
// WIN64-LABEL: void @x87_after_f64x2(ptr {{.*}}sret([16 x i8]) align 16 {{.*}}, ptr {{.*}})
#[unsafe(no_mangle)]
extern "C" fn x87_after_f64x2(x: X87After<f64x2>) -> X87After<f64x2> {
    x
}
