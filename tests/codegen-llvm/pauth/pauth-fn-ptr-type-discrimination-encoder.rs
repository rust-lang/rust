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
// The `encode_ty` function in <compiler/rustc_middle/src/ptrauth/discriminator.rs> is responsible
// for converting types to the literal values that are then used as the basis for hashing. Its
// implementation is a faithful translation of Clang's `encodeTypeForFunctionPointerAuth`.

#![feature(repr_simd)]
#![feature(simd_ffi)]
#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::{c_void, mem};

// Builtin types.
extern "C" {
    fn f_i32(x: i32) -> i32;
    fn f_f(x: f32) -> f32;
    fn f_d(x: f64) -> f64;
    fn f_2d(x: f64, y: f64) -> f64;
    fn f_ld(x: f64) -> f64;
    fn f_v() -> ();
}
type fn_i32 = unsafe extern "C" fn(i32) -> i32;
type fn_f = unsafe extern "C" fn(f32) -> f32;
type fn_d = unsafe extern "C" fn(f64) -> f64;
type fn_2d = unsafe extern "C" fn(f64, f64) -> f64;
type fn_v = unsafe extern "C" fn() -> ();
// discriminator: 2981 (0x0BA5), encoding: FiiE
// DISC: @{{.*}}T_I32 = constant ptr ptrauth (ptr @f_i32, i32 0, i64 2981), align 8
// NO_DISC: @{{.*}}T_I32 = constant ptr ptrauth (ptr @f_i32, i32 0), align 8
#[used]
static T_I32: fn_i32 = f_i32;
// discriminator: 28450 (0x6F22), encoding: FffE
// DISC: @{{.*}}T_F = constant ptr ptrauth (ptr @f_f, i32 0, i64 28450), align 8
// NO_DISC: @{{.*}}T_F = constant ptr ptrauth (ptr @f_f, i32 0), align 8
#[used]
static T_F: fn_f = f_f;
// discriminator: 43115 (0xA86B), encoding: FddE
// DISC: @{{.*}}T_D = constant ptr ptrauth (ptr @f_d, i32 0, i64 43115), align 8
// NO_DISC: @{{.*}}T_D = constant ptr ptrauth (ptr @f_d, i32 0), align 8
#[used]
static T_D: fn_d = f_d;
// discriminator: 38695 (0x9727), encoding: FdddE
// DISC: @{{.*}}T_2D = constant ptr ptrauth (ptr @f_2d, i32 0, i64 38695), align 8
// NO_DISC: @{{.*}}T_2D = constant ptr ptrauth (ptr @f_2d, i32 0), align 8
#[used]
static T_2D: fn_2d = f_2d;
// discriminator: 18983 (0x4A27), encoding: FvE
// DISC: @{{.*}}T_V = constant ptr ptrauth (ptr @f_v, i32 0, i64 18983), align 8
// NO_DISC: @{{.*}}T_V = constant ptr ptrauth (ptr @f_v, i32 0), align 8
#[used]
static T_V: fn_v = f_v;

// Pointer types.
extern "C" {
    fn f_ptr(x: *mut i32) -> i32;
}
type fn_ptr = unsafe extern "C" fn(*mut i32) -> i32;
// discriminator: 12410 (0x307A), encoding: FiPE
// DISC: @{{.*}}T_PTR = constant ptr ptrauth (ptr @f_ptr, i32 0, i64 12410), align 8
// NO_DISC: @{{.*}}T_PTR = constant ptr ptrauth (ptr @f_ptr, i32 0), align 8
#[used]
static T_PTR: fn_ptr = f_ptr;

// Array types.
extern "C" {
    fn f_arr_2(x: *mut i32) -> i32;
    fn f_arr_4(x: *mut i32) -> i32;
    fn f_arr2(x: *mut i32) -> i32;
}
type fn_arr_2 = unsafe extern "C" fn(*mut i32) -> i32;
type fn_arr_4 = unsafe extern "C" fn(*mut i32) -> i32;
type fn_arr2 = unsafe extern "C" fn(*mut i32) -> i32;
// discriminator: 12410 (0x307A), encoding: FiPE
// DISC: @{{.*}}T_ARR_2 = constant ptr ptrauth (ptr @f_arr_2, i32 0, i64 12410), align 8
// NO_DISC: @{{.*}}T_ARR_2 = constant ptr ptrauth (ptr @f_arr_2, i32 0), align 8
#[used]
static T_ARR_2: fn_arr_2 = f_arr_2;
// discriminator: 12410 (0x307A), encoding: FiPE
// DISC: @{{.*}}T_ARR_4 = constant ptr ptrauth (ptr @f_arr_4, i32 0, i64 12410), align 8
// NO_DISC: @{{.*}}T_ARR_4 = constant ptr ptrauth (ptr @f_arr_4, i32 0), align 8
#[used]
static T_ARR_4: fn_arr_4 = f_arr_4;
// discriminator: 12410 (0x307A), encoding: FiPE
// DISC: @{{.*}}T_ARR2 = constant ptr ptrauth (ptr @f_arr2, i32 0, i64 12410), align 8
// NO_DISC: @{{.*}}T_ARR2 = constant ptr ptrauth (ptr @f_arr2, i32 0), align 8
#[used]
static T_ARR2: fn_arr2 = f_arr2;

// Function types.
extern "C" {
    fn f_nested(g: extern "C" fn(i32) -> i32, x: i32) -> i32;
}
type fn_i32_i32 = extern "C" fn(i32) -> i32;
type fn_nested = unsafe extern "C" fn(fn_i32_i32, i32) -> i32;
// discriminator: 20679 (0x50C7), encoding: FiPiE
// DISC: @{{.*}}T_NESTED = constant ptr ptrauth (ptr @f_nested, i32 0, i64 20679), align 8
// NO_DISC: @{{.*}}T_NESTED = constant ptr ptrauth (ptr @f_nested, i32 0), align 8
#[used]
static T_NESTED: fn_nested = f_nested;

// Variadic function.
extern "C" {
    fn f_var(x: i32, ...) -> i32;
}
type fn_var = unsafe extern "C" fn(i32, ...) -> i32;
// discriminator: 7476 (0x1D34), encoding: FiizE
// DISC: @{{.*}}T_VAR = constant ptr ptrauth (ptr @f_var, i32 0, i64 7476), align 8
// NO_DISC: @{{.*}}T_VAR = constant ptr ptrauth (ptr @f_var, i32 0), align 8
#[used]
static T_VAR: fn_var = f_var;

// Enum coercion to int.
#[repr(i32)]
enum MyEnum {
    A = 1,
    B = 2,
}
extern "C" {
    fn f_enum(x: MyEnum) -> MyEnum;
}
type fn_enum = unsafe extern "C" fn(MyEnum) -> MyEnum;
// discriminator: 2981 (0x0BA5), encoding: FiiE
// DISC: @{{.*}}T_ENUM = constant ptr ptrauth (ptr @f_enum, i32 0, i64 2981), align 8
// NO_DISC: @{{.*}}T_ENUM = constant ptr ptrauth (ptr @f_enum, i32 0), align 8
#[used]
static T_ENUM: fn_enum = f_enum;

// Struct types.
#[repr(C)]
struct MyStruct {
    x: i32,
}
extern "C" {
    fn f_struct(x: MyStruct) -> MyStruct;
}
type fn_struct = unsafe extern "C" fn(MyStruct) -> MyStruct;
// discriminator: 17754 (0x455A), encoding: F8MyStruct8MyStructE
// DISC: @{{.*}}T_STRUCT = constant ptr ptrauth (ptr @f_struct, i32 0, i64 17754), align 8
// NO_DISC: @{{.*}}T_STRUCT = constant ptr ptrauth (ptr @f_struct, i32 0), align 8
#[used]
static T_STRUCT: fn_struct = f_struct;

// Function pointer as arguments.
extern "C" {
    fn f_fp(h: extern "C" fn(i32) -> i32) -> i32;
}
type fn_i32_i32_fp_as_arg = extern "C" fn(i32) -> i32;
type fn_fp = unsafe extern "C" fn(fn_i32_i32_fp_as_arg) -> i32;
// discriminator: 12410 (0x307A), encoding: FiPE
// DISC: @{{.*}}T_FP = constant ptr ptrauth (ptr @f_fp, i32 0, i64 12410), align 8
// NO_DISC: @{{.*}}T_FP = constant ptr ptrauth (ptr @f_fp, i32 0), align 8
#[used]
static T_FP: fn_fp = f_fp;

// SIMD vector type.
#[repr(simd)]
struct Int4([i32; 4]);
extern "C" {
    fn f_vec(x: Int4) -> Int4;
}
type FnVec = unsafe extern "C" fn(Int4) -> Int4;
// discriminator: 34246 (0x85C6), encoding: FDv16Dv16E
// DISC: @{{.*}}T_VEC = constant ptr ptrauth (ptr @f_vec, i32 0, i64 34246), align 8
// NO_DISC: @{{.*}}T_VEC = constant ptr ptrauth (ptr @f_vec, i32 0), align 8
#[used]
static T_VEC: FnVec = f_vec;

// Mixed.
type fn_i32_f = unsafe extern "C" fn(f32) -> i32;
extern "C" {
    fn f_mixed(g: fn_i32_f, arr: *mut *mut f32, d: f64) -> i32;
}
type fn_mixed = unsafe extern "C" fn(fn_i32_f, *mut *mut f32, f64) -> i32;
// discriminator: 36791 (0x8FB7), encoding: FiPPdE
// DISC: @{{.*}}T_MIXED = constant ptr ptrauth (ptr @f_mixed, i32 0, i64 36791), align 8
// NO_DISC: @{{.*}}T_MIXED = constant ptr ptrauth (ptr @f_mixed, i32 0), align 8
#[used]
static T_MIXED: fn_mixed = f_mixed;

// Quicksort.
type FnCmp = unsafe extern "C" fn(*const c_void, *const c_void) -> i32;
type FnQsort = unsafe extern "C" fn(*mut c_void, usize, usize, FnCmp);
extern "C" {
    fn quickSort(base: *mut c_void, n: usize, size: usize, cmp: FnCmp);

    fn cmpI32Ascending(lhs: *const c_void, rhs: *const c_void) -> i32;
}
#[used]
static T_QSORT: FnQsort = quickSort;
// discriminator: 39926 (0x9BF6) of: FvPiiPE
// DISC: @{{.*}}T_QSORT = constant ptr ptrauth (ptr @quickSort, i32 0, i64 39926), align 8
// NO_DISC: @{{.*}}T_QSORT = constant ptr ptrauth (ptr @quickSort, i32 0), align 8
#[used]
// discriminator: 58622 (0xE4FE) of: FiPPE
// DISC: @{{.*}}T_CMP_I32_ASCENDING = constant ptr ptrauth (ptr @cmpI32Ascending, i32 0, i64 58622), align 8
// NO_DISC: @{{.*}}T_CMP_I32_ASCENDING = constant ptr ptrauth (ptr @cmpI32Ascending, i32 0), align 8
static T_CMP_I32_ASCENDING: FnCmp = cmpI32Ascending;

// Callbacks.
extern "C" fn callback_i32(x: i32) -> i32 {
    x + 1
}
unsafe extern "C" fn callback_f32_to_i32(x: f32) -> i32 {
    x as i32
}
type FnCallbackI32 = unsafe extern "C" fn(i32) -> i32;
type FnCallbackF32ToI32 = unsafe extern "C" fn(f32) -> i32;
#[used]
// discriminator: 2981 (0x0BA5) of: FiiE
// DISC: @{{.*}}T_CALLBACK_I32 = constant ptr ptrauth (ptr @{{.*}}callback_i32, i32 0, i64 2981), align 8
// NO_DISC: @{{.*}}T_CALLBACK_I32 = constant ptr ptrauth (ptr @{{.*}}callback_i32, i32 0), align 8
static T_CALLBACK_I32: FnCallbackI32 = callback_i32;
#[used]
// discriminator: 48468 (0xBD54) of: FifE
// DISC: @{{.*}}T_CALLBACK_F32_TO_I32 = constant ptr ptrauth (ptr @{{.*}}callback_f32_to_i32, i32 0, i64 48468), align 8
// NO_DISC: @{{.*}}T_CALLBACK_F32_TO_I32 = constant ptr ptrauth (ptr @{{.*}}callback_f32_to_i32, i32 0), align 8
static T_CALLBACK_F32_TO_I32: FnCallbackF32ToI32 = callback_f32_to_i32;

// Test the calling of the functions.
pub fn main() {
    unsafe {
        // Builtin types.

        // DISC: %{{.*}} = call i32 ptrauth (ptr @f_i32, i32 0, i64 2981)(i32 123) {{.*}} [ "ptrauth"(i32 0, i64 2981) ]
        // NO_DISC: %{{.*}} = call i32 ptrauth (ptr @f_i32, i32 0)(i32 123) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_I32(123);
        // DISC: %{{.*}} = call float ptrauth (ptr @f_f, i32 0, i64 28450)(float 1.250000e+00) {{.*}} [ "ptrauth"(i32 0, i64 28450) ]
        // NO_DISC: %{{.*}} = call float ptrauth (ptr @f_f, i32 0)(float 1.250000e+00) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_F(1.25);
        // DISC: %{{.*}} = call double ptrauth (ptr @f_d, i32 0, i64 43115)(double 2.500000e+00) {{.*}} [ "ptrauth"(i32 0, i64 43115) ]
        // NO_DISC: %{{.*}} = call double ptrauth (ptr @f_d, i32 0)(double 2.500000e+00) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_D(2.5);
        // DISC: %{{.*}} = call double ptrauth (ptr @f_2d, i32 0, i64 38695)(double 1.000000e+00, double 2.000000e+00) {{.*}} [ "ptrauth"(i32 0, i64 38695) ]
        // NO_DISC: %{{.*}} = call double ptrauth (ptr @f_2d, i32 0)(double 1.000000e+00, double 2.000000e+00) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_2D(1.0, 2.0);
        // DISC: call void ptrauth (ptr @f_v, i32 0, i64 18983)() {{.*}} [ "ptrauth"(i32 0, i64 18983) ]
        // NO_DISC: call void ptrauth (ptr @f_v, i32 0)() {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        T_V();

        // Pointer type.
        let mut x = 42i32;
        // DISC: %{{.*}} = call i32 ptrauth (ptr @f_ptr, i32 0, i64 12410)(ptr %x) {{.*}} [ "ptrauth"(i32 0, i64 12410) ]
        // NO_DISC: %{{.*}} = call i32 ptrauth (ptr @f_ptr, i32 0)(ptr %x) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_PTR(&mut x);

        // Array types.
        let mut arr2 = [1i32, 2];
        let mut arr4 = [1i32, 2, 3, 4];
        let mut arrn = [1i32, 2, 3];

        // DISC-DAG: call i32 ptrauth (ptr @f_arr_2, i32 0, i64 12410)(ptr %arr2) {{.*}} [ "ptrauth"(i32 0, i64 12410) ]
        // NO_DISC-DAG: call i32 ptrauth (ptr @f_arr_2, i32 0)(ptr %arr2) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_ARR_2((&mut arr2) as *mut [i32; 2] as *mut i32);
        // DISC-DAG: call i32 ptrauth (ptr @f_arr_4, i32 0, i64 12410)(ptr %arr4) {{.*}} [ "ptrauth"(i32 0, i64 12410) ]
        // NO_DISC-DAG: call i32 ptrauth (ptr @f_arr_4, i32 0)(ptr %arr4) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_ARR_4((&mut arr4) as *mut [i32; 4] as *mut i32);
        // DISC-DAG: call i32 ptrauth (ptr @f_arr2, i32 0, i64 12410)(ptr %arrn) {{.*}} [ "ptrauth"(i32 0, i64 12410) ]
        // NO_DISC-DAG: call i32 ptrauth (ptr @f_arr2, i32 0)(ptr %arrn) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_ARR2((&mut arrn) as *mut [i32; 3] as *mut i32);

        // Function argument.
        // DISC: call i32 ptrauth (ptr @f_nested, i32 0, i64 20679)(ptr ptrauth (ptr @{{.*}}callback_i32, i32 0, i64 2981), i32 123) {{.*}} [ "ptrauth"(i32 0, i64 20679) ]
        // NO_DISC: call i32 ptrauth (ptr @f_nested, i32 0)(ptr ptrauth (ptr @{{.*}}callback_i32, i32 0), i32 123) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_NESTED(callback_i32, 123);

        // Variadic.
        // DISC: call i32 (i32, ...) ptrauth (ptr @f_var, i32 0, i64 7476){{.*}} [ "ptrauth"(i32 0, i64 7476) ]
        // NO_DISC: call i32 (i32, ...) ptrauth (ptr @f_var, i32 0){{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_VAR(3, 10i32, 20i32, 30i32);

        // Enum.
        // DISC: call i32 ptrauth (ptr @f_enum, i32 0, i64 2981)(i32 1) {{.*}} [ "ptrauth"(i32 0, i64 2981) ]
        // NO_DISC: call i32 ptrauth (ptr @f_enum, i32 0)(i32 1) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_ENUM(MyEnum::A);

        // Struct.
        // DISC: call i64 ptrauth (ptr @f_struct, i32 0, i64 17754){{.*}} [ "ptrauth"(i32 0, i64 17754) ]
        // NO_DISC: ){{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_STRUCT(MyStruct { x: 123 });

        // Function pointer argument.
        // DISC: call i32 ptrauth (ptr @f_fp, i32 0, i64 12410)(ptr ptrauth (ptr @{{.*}}callback_i32, i32 0, i64 2981)) {{.*}} [ "ptrauth"(i32 0, i64 12410) ]
        // NO_DISC: call i32 ptrauth (ptr @f_fp, i32 0)(ptr ptrauth (ptr @{{.*}}callback_i32, i32 0)) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_FP(callback_i32);

        // SIMD vector.
        // DISC: call <4 x i32> ptrauth (ptr @f_vec, i32 0, i64 34246)(<4 x i32>{{.*}} [ "ptrauth"(i32 0, i64 34246) ]
        // NO_DISC: call <4 x i32> ptrauth (ptr @f_vec, i32 0)(<4 x i32>{{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_VEC(Int4([1, 2, 3, 4]));

        // Mixed case.
        let mut value = 1.0f32;
        let mut ptr = &mut value as *mut f32;
        // DISC: call i32 ptrauth (ptr @f_mixed, i32 0, i64 36791)(ptr ptrauth (ptr @{{.*}}callback_f32_to_i32, i32 0, i64 48468), {{.*}} [ "ptrauth"(i32 0, i64 36791) ]
        // NO_DISC: call i32 ptrauth (ptr @f_mixed, i32 0)(ptr ptrauth (ptr @{{.*}}callback_f32_to_i32, i32 0), {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_MIXED(callback_f32_to_i32, &mut ptr, 2.0);

        // Comparator.
        let lhs = 1i32;
        let rhs = 2i32;
        // DISC: call i32 ptrauth (ptr @cmpI32Ascending, i32 0, i64 58622)(ptr %lhs, ptr %rhs) {{.*}} [ "ptrauth"(i32 0, i64 58622) ]
        // NO_DISC: call i32 ptrauth (ptr @cmpI32Ascending, i32 0)(ptr %lhs, ptr %rhs) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let _ = T_CMP_I32_ASCENDING(
            // (&lhs as *const i32).cast(),
            // (&rhs as *const i32).cast(),
            (&lhs as *const i32) as *const c_void,
            (&rhs as *const i32) as *const c_void,
        );

        // Quicksort.
        let mut values = [42i32, 7, 19, 3, 11];
        // DISC: call void ptrauth (ptr @quickSort, i32 0, i64 39926)(ptr %values, i64 5, i64 4, ptr ptrauth (ptr @cmpI32Ascending, i32 0, i64 58622)) {{.*}} [ "ptrauth"(i32 0, i64 39926) ]
        // NO_DISC: call void ptrauth (ptr @quickSort, i32 0)(ptr %values, i64 5, i64 4, ptr ptrauth (ptr @cmpI32Ascending, i32 0)) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        T_QSORT(
            (&mut values as *mut [i32; 5]) as *mut i32 as *mut c_void,
            //values.as_mut_ptr().cast(),
            5,
            4,
            T_CMP_I32_ASCENDING,
        );
    }
}

// Equivalent C code:
//
// #include <stdint.h>
// #include <stdlib.h>
//
// // Builtin types.
// int32_t f_i32(int32_t x);
// float f_f(float x);
// double f_d(double x);
// double f_2d(double x, double y);
// long double f_ld(long double x);
// void f_v(void);
//
// typedef int32_t (*fn_i32)(int32_t);
// typedef float (*fn_f)(float);
// typedef double (*fn_d)(double);
// typedef double (*fn_2d)(double, double);
// typedef void (*fn_v)(void);
//
// __attribute__((used)) static fn_i32 T_I32 = f_i32;
// __attribute__((used)) static fn_f T_F = f_f;
// __attribute__((used)) static fn_d T_D = f_d;
// __attribute__((used)) static fn_2d T_2D = f_2d;
// __attribute__((used)) static fn_v T_V = f_v;
//
// // Pointer types.
// int32_t f_ptr(int32_t *x);
//
// typedef int32_t (*fn_ptr)(int32_t *);
//
// __attribute__((used)) static fn_ptr T_PTR = f_ptr;
//
// // Array types.
// int32_t f_arr_2(int32_t x[2]);
// int32_t f_arr_4(int32_t x[4]);
//
// typedef int32_t (*fn_arr_2)(int32_t[2]);
// typedef int32_t (*fn_arr_4)(int32_t[4]);
//
// __attribute__((used)) static fn_arr_2 T_ARR_2 = f_arr_2;
// __attribute__((used)) static fn_arr_4 T_ARR_4 = f_arr_4;
// // incomplete array
// int32_t f_arr2(int32_t x[]);
//
// typedef int32_t (*fn_arr2)(int32_t[]);
//
// __attribute__((used)) static fn_arr2 T_ARR2 = f_arr2;
//
// // Function types.
// int32_t f_nested(int32_t (*g)(int32_t), int32_t x);
//
// typedef int32_t (*fn_i32_i32)(int32_t);
// typedef int32_t (*fn_nested)(fn_i32_i32, int32_t);
//
// __attribute__((used)) static fn_nested T_NESTED = f_nested;
//
// // Variadic function.
// int32_t f_var(int32_t x, ...);
//
// typedef int32_t (*fn_var)(int32_t, ...);
//
// __attribute__((used)) static fn_var T_VAR = f_var;
//
// // Enum to integer coercion.
// typedef enum { A = 1, B = 2 } MyEnum;
//
// MyEnum f_enum(MyEnum x);
//
// typedef MyEnum (*fn_enum)(MyEnum);
//
// __attribute__((used)) static fn_enum T_ENUM = f_enum;
//
// // Struct.
// typedef struct {
//   int x;
// } MyStruct;
//
// MyStruct f_struct(MyStruct x);
//
// typedef MyStruct (*fn_struct)(MyStruct);
//
// __attribute__((used)) static fn_struct T_STRUCT = f_struct;
//
// // Pointer to function pointer.
// int32_t f_fp(int32_t (*h)(int32_t));
//
// typedef int32_t (*fn_i32_i32)(int32_t);
// typedef int32_t (*fn_fp)(fn_i32_i32);
//
// __attribute__((used)) static fn_fp T_FP = f_fp;
//
// // SIMD vector.
// typedef int32_t int4 __attribute__((vector_size(16)));
//
// int4 f_vec(int4 x);
//
// typedef int4 (*fn_vec)(int4);
//
// __attribute__((used)) static fn_vec T_VEC = f_vec;
//
// // Mix.
// int32_t f_mixed(int32_t (*g)(float), float *arr[4], double d);
//
// typedef int32_t (*fn_mixed)(int32_t (*)(float), float *[4], double);
//
// __attribute__((used)) static fn_mixed T_MIXED = f_mixed;
//
// // Qsort
// void quickSort(void *Base, size_t N, size_t Size,
//                int (*Cmp)(const void *, const void *));
//
// int cmpI32Ascending(const void *LHS, const void *RHS);
// typedef void (*fn_qsort)(void *, size_t, size_t,
//                          int (*)(const void *, const void *));
// typedef int (*fn_cmp)(const void *, const void *);
//
// __attribute__((used)) static fn_qsort T_QSORT = quickSort;
// __attribute__((used)) static fn_cmp T_CMP_I32_ASCENDING = cmpI32Ascending;
//
// // Callbacks
// static int32_t callback_i32(int32_t x) { return x + 1; }
// static int32_t callback_f32_to_i32(float x) { return (int32_t)x; }
// typedef int32_t (*fn_callback_i32)(int32_t);
// typedef int32_t (*fn_callback_f32_to_i32)(float);
// __attribute__((used)) static fn_callback_i32 T_CALLBACK_I32 = callback_i32;
// __attribute__((used)) static fn_callback_f32_to_i32 T_CALLBACK_F32_TO_I32 =
//     callback_f32_to_i32;
//
// int main(void) {
//   /* Builtin types. */
//   (void)T_I32(123);
//   (void)T_F(1.25f);
//   (void)T_D(2.5);
//   (void)T_2D(1.0, 2.0);
//   T_V();
//
//   /* Pointer type. */
//   int32_t x = 42;
//   (void)T_PTR(&x);
//
//   /* Array types. */
//   int32_t arr2[2] = {1, 2};
//   int32_t arr4[4] = {1, 2, 3, 4};
//   int32_t arrn[3] = {1, 2, 3};
//
//   (void)T_ARR_2(arr2);
//   (void)T_ARR_4(arr4);
//   (void)T_ARR2(arrn);
//
//   /* Function argument. */
//   (void)T_NESTED(callback_i32, 123);
//
//   /* Variadic. */
//   (void)T_VAR(3, 10, 20, 30);
//
//   /* Enum. */
//   (void)T_ENUM(A);
//
//   /* Struct. */
//   (void)T_STRUCT((MyStruct){.x = 123});
//
//   /* Function pointer argument. */
//   (void)T_FP(callback_i32);
//
//   /* SIMD vector. */
//   int4 v = {1, 2, 3, 4};
//   (void)T_VEC(v);
//
//   /* Mixed case. */
//   float value0 = 1.0f;
//   float value1 = 2.0f;
//   float value2 = 3.0f;
//   float value3 = 4.0f;
//
//   float *arrp[4] = {&value0, &value1, &value2, &value3};
//
//   (void)T_MIXED(callback_f32_to_i32, arrp, 1.0);
//
//   /* Comparator. */
//   int32_t lhs = 1;
//   int32_t rhs = 2;
//
//   (void)T_CMP_I32_ASCENDING(&lhs, &rhs);
//
//   /* Quicksort. */
//   int32_t values[] = {42, 7, 19, 3, 11};
//
//   T_QSORT(values, sizeof(values) / sizeof(values[0]), sizeof(values[0]),
//           T_CMP_I32_ASCENDING);
//
//   return 0;
// }
