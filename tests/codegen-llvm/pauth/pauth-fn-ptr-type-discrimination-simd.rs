// ignore-tidy-file-linelength
//@ add-minicore
//@ only-pauthtest
// Run it at O0, so that the compiler doesn't optimise the calls away.

//@ revisions: DISC NO_DISC
//@ [DISC] needs-llvm-components: aarch64
//@ [DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=+function-pointer-type-discrimination -C opt-level=0
//@ [NO_DISC] needs-llvm-components: aarch64
//@ [NO_DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=-function-pointer-type-discrimination -C opt-level=0

// Test generation of function-pointer type discriminators. The discriminator values were obtained
// from Clang by compiling equivalent C code (included). Both compilers must generate identical
// values.
//
// For encoding purposes, Clang is only interested in the total size of the vector, so all the
// combinations below should generate the same encoding: FDv16Dv16E, discriminator: 34246 (0x85C6).

#![feature(no_core, lang_items, repr_simd, simd_ffi)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;

#[repr(simd)]
struct I8x16([i8; 16]);
#[repr(simd)]
struct I16x8([i16; 8]);
#[repr(simd)]
struct I32x4([i32; 4]);
#[repr(simd)]
struct I64x2([i64; 2]);
#[repr(simd)]
struct U8x16([u8; 16]);
#[repr(simd)]
struct U16x8([u16; 8]);
#[repr(simd)]
struct U32x4([u32; 4]);
#[repr(simd)]
struct U64x2([u64; 2]);
#[repr(simd)]
struct F32x4([f32; 4]);
#[repr(simd)]
struct F64x2([f64; 2]);

extern "C" {
    fn f_i8x16(x: I8x16) -> I8x16;
    fn f_i16x8(x: I16x8) -> I16x8;
    fn f_i32x4(x: I32x4) -> I32x4;
    fn f_i64x2(x: I64x2) -> I64x2;
    fn f_u8x16(x: U8x16) -> U8x16;
    fn f_u16x8(x: U16x8) -> U16x8;
    fn f_u32x4(x: U32x4) -> U32x4;
    fn f_u64x2(x: U64x2) -> U64x2;
    fn f_f32x4(x: F32x4) -> F32x4;
    fn f_f64x2(x: F64x2) -> F64x2;
}

type FnI8x16 = unsafe extern "C" fn(I8x16) -> I8x16;
type FnI16x8 = unsafe extern "C" fn(I16x8) -> I16x8;
type FnI32x4 = unsafe extern "C" fn(I32x4) -> I32x4;
type FnI64x2 = unsafe extern "C" fn(I64x2) -> I64x2;
type FnU8x16 = unsafe extern "C" fn(U8x16) -> U8x16;
type FnU16x8 = unsafe extern "C" fn(U16x8) -> U16x8;
type FnU32x4 = unsafe extern "C" fn(U32x4) -> U32x4;
type FnU64x2 = unsafe extern "C" fn(U64x2) -> U64x2;
type FnF32x4 = unsafe extern "C" fn(F32x4) -> F32x4;
type FnF64x2 = unsafe extern "C" fn(F64x2) -> F64x2;

#[used]
// DISC: {{.*}}T_I8x16 = constant ptr ptrauth (ptr @f_i8x16, i32 0, i64 34246)
// NO_DISC: {{.*}}T_I8x16 = constant ptr ptrauth (ptr @f_i8x16, i32 0)
static T_I8x16: FnI8x16 = f_i8x16;
#[used]
// DISC: {{.*}}T_I16x8 = constant ptr ptrauth (ptr @f_i16x8, i32 0, i64 34246)
// NO_DISC: {{.*}}T_I16x8 = constant ptr ptrauth (ptr @f_i16x8, i32 0)
static T_I16x8: FnI16x8 = f_i16x8;
#[used]
// DISC: {{.*}}T_I32x4 = constant ptr ptrauth (ptr @f_i32x4, i32 0, i64 34246)
// NO_DISC: {{.*}}T_I32x4 = constant ptr ptrauth (ptr @f_i32x4, i32 0)
static T_I32x4: FnI32x4 = f_i32x4;
#[used]
// DISC: {{.*}}T_I64x2 = constant ptr ptrauth (ptr @f_i64x2, i32 0, i64 34246)
// NO_DISC: {{.*}}T_I64x2 = constant ptr ptrauth (ptr @f_i64x2, i32 0)
static T_I64x2: FnI64x2 = f_i64x2;
#[used]
// DISC: {{.*}}T_U8x16 = constant ptr ptrauth (ptr @f_u8x16, i32 0, i64 34246)
// NO_DISC: {{.*}}T_U8x16 = constant ptr ptrauth (ptr @f_u8x16, i32 0)
static T_U8x16: FnU8x16 = f_u8x16;
#[used]
// DISC: {{.*}}T_U16x8 = constant ptr ptrauth (ptr @f_u16x8, i32 0, i64 34246)
// NO_DISC: {{.*}}T_U16x8 = constant ptr ptrauth (ptr @f_u16x8, i32 0)
static T_U16x8: FnU16x8 = f_u16x8;
#[used]
// DISC: {{.*}}T_U32x4 = constant ptr ptrauth (ptr @f_u32x4, i32 0, i64 34246)
// NO_DISC: {{.*}}T_U32x4 = constant ptr ptrauth (ptr @f_u32x4, i32 0)
static T_U32x4: FnU32x4 = f_u32x4;
#[used]
// DISC: {{.*}}T_U64x2 = constant ptr ptrauth (ptr @f_u64x2, i32 0, i64 34246)
// NO_DISC: {{.*}}T_U64x2 = constant ptr ptrauth (ptr @f_u64x2, i32 0)
static T_U64x2: FnU64x2 = f_u64x2;
#[used]
// DISC: {{.*}}T_F32x4 = constant ptr ptrauth (ptr @f_f32x4, i32 0, i64 34246)
// NO_DISC: {{.*}}T_F32x4 = constant ptr ptrauth (ptr @f_f32x4, i32 0)
static T_F32x4: FnF32x4 = f_f32x4;
#[used]
// DISC: {{.*}}T_F64x2 = constant ptr ptrauth (ptr @f_f64x2, i32 0, i64 34246)
// NO_DISC: {{.*}}T_F64x2 = constant ptr ptrauth (ptr @f_f64x2, i32 0)
static T_F64x2: FnF64x2 = f_f64x2;

pub fn main() {
    unsafe {
        // DISC: call <16 x i8> ptrauth (ptr @f_i8x16, i32 0, i64 34246)(<16 x i8> %{{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 34246) ]
        // NO_DISC: call <16 x i8> ptrauth (ptr @f_i8x16, i32 0)(<16 x i8> %{{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_I8x16(I8x16([0; 16]));
        // DISC: call <8 x i16> ptrauth (ptr @f_i16x8, i32 0, i64 34246)(<8 x i16> %{{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 34246) ]
        // NO_DISC: call <8 x i16> ptrauth (ptr @f_i16x8, i32 0)(<8 x i16> %{{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_I16x8(I16x8([0; 8]));
        // DISC: call <4 x i32> ptrauth (ptr @f_i32x4, i32 0, i64 34246)(<4 x i32> %{{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 34246) ]
        // NO_DISC: call <4 x i32> ptrauth (ptr @f_i32x4, i32 0)(<4 x i32> %{{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_I32x4(I32x4([0; 4]));
        // DISC: call <2 x i64> ptrauth (ptr @f_i64x2, i32 0, i64 34246)(<2 x i64> %{{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 34246) ]
        // NO_DISC: call <2 x i64> ptrauth (ptr @f_i64x2, i32 0)(<2 x i64> %{{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_I64x2(I64x2([0; 2]));
        // DISC: call <16 x i8> ptrauth (ptr @f_u8x16, i32 0, i64 34246)(<16 x i8> %{{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 34246) ]
        // NO_DISC: call <16 x i8> ptrauth (ptr @f_u8x16, i32 0)(<16 x i8> %{{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_U8x16(U8x16([0; 16]));
        // DISC: call <8 x i16> ptrauth (ptr @f_u16x8, i32 0, i64 34246)(<8 x i16> %{{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 34246) ]
        // NO_DISC: call <8 x i16> ptrauth (ptr @f_u16x8, i32 0)(<8 x i16> %{{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_U16x8(U16x8([0; 8]));
        // DISC: call <4 x i32> ptrauth (ptr @f_u32x4, i32 0, i64 34246)(<4 x i32> %{{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 34246) ]
        // NO_DISC: call <4 x i32> ptrauth (ptr @f_u32x4, i32 0)(<4 x i32> %{{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_U32x4(U32x4([0; 4]));
        // DISC: call <2 x i64> ptrauth (ptr @f_u64x2, i32 0, i64 34246)(<2 x i64> %{{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 34246) ]
        // NO_DISC: call <2 x i64> ptrauth (ptr @f_u64x2, i32 0)(<2 x i64> %{{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_U64x2(U64x2([0; 2]));
        // DISC: call <4 x float> ptrauth (ptr @f_f32x4, i32 0, i64 34246)(<4 x float> %{{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 34246) ]
        // NO_DISC: call <4 x float> ptrauth (ptr @f_f32x4, i32 0)(<4 x float> %{{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_F32x4(F32x4([0.0; 4]));
        // DISC: call <2 x double> ptrauth (ptr @f_f64x2, i32 0, i64 34246)(<2 x double> %{{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 34246) ]
        // NO_DISC: call <2 x double> ptrauth (ptr @f_f64x2, i32 0)(<2 x double> %{{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_F64x2(F64x2([0.0; 2]));
    }
}

// Equivalent C sample:
//
// ```c
// typedef signed char I8x16 __attribute__((vector_size(16)));
// typedef short I16x8 __attribute__((vector_size(16)));
// typedef int I32x4 __attribute__((vector_size(16)));
// typedef long long I64x2 __attribute__((vector_size(16)));
//
// typedef unsigned char U8x16 __attribute__((vector_size(16)));
// typedef unsigned short U16x8 __attribute__((vector_size(16)));
// typedef unsigned int U32x4 __attribute__((vector_size(16)));
// typedef unsigned long long U64x2 __attribute__((vector_size(16)));
//
// typedef float F32x4 __attribute__((vector_size(16)));
// typedef double F64x2 __attribute__((vector_size(16)));
//
// extern I8x16 f_i8x16(I8x16);
// extern I16x8 f_i16x8(I16x8);
// extern I32x4 f_i32x4(I32x4);
// extern I64x2 f_i64x2(I64x2);
// extern U8x16 f_u8x16(U8x16);
// extern U16x8 f_u16x8(U16x8);
// extern U32x4 f_u32x4(U32x4);
// extern U64x2 f_u64x2(U64x2);
// extern F32x4 f_f32x4(F32x4);
// extern F64x2 f_f64x2(F64x2);
//
// typedef I8x16 (*FnI8x16)(I8x16);
// typedef I16x8 (*FnI16x8)(I16x8);
// typedef I32x4 (*FnI32x4)(I32x4);
// typedef I64x2 (*FnI64x2)(I64x2);
// typedef U8x16 (*FnU8x16)(U8x16);
// typedef U16x8 (*FnU16x8)(U16x8);
// typedef U32x4 (*FnU32x4)(U32x4);
// typedef U64x2 (*FnU64x2)(U64x2);
// typedef F32x4 (*FnF32x4)(F32x4);
// typedef F64x2 (*FnF64x2)(F64x2);
//
// FnI8x16 T_I8x16 = f_i8x16;
// FnI16x8 T_I16x8 = f_i16x8;
// FnI32x4 T_I32x4 = f_i32x4;
// FnI64x2 T_I64x2 = f_i64x2;
// FnU8x16 T_U8x16 = f_u8x16;
// FnU16x8 T_U16x8 = f_u16x8;
// FnU32x4 T_U32x4 = f_u32x4;
// FnU64x2 T_U64x2 = f_u64x2;
// FnF32x4 T_F32x4 = f_f32x4;
// FnF64x2 T_F64x2 = f_f64x2;
//
// void test(void) {
//   T_I8x16((I8x16){0});
//   T_I16x8((I16x8){0});
//   T_I32x4((I32x4){0});
//   T_I64x2((I64x2){0});
//   T_U8x16((U8x16){0});
//   T_U16x8((U16x8){0});
//   T_U32x4((U32x4){0});
//   T_U64x2((U64x2){0});
//   T_F32x4((F32x4){0.0f});
//   T_F64x2((F64x2){0.0});
// }
// ```
