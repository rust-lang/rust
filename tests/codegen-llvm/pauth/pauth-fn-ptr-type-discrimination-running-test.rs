// ignore-tidy-file-linelength
//@ add-minicore
//@ only-pauthtest
// Run it at O0, so that the compiler doesn't optimise the calls away.

//@ revisions: DISC NO_DISC
//@ [DISC] needs-llvm-components: aarch64
//@ [DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=+function-pointer-type-discrimination -C opt-level=0
//@ [NO_DISC] needs-llvm-components: aarch64
//@ [NO_DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=-function-pointer-type-discrimination -C opt-level=0

// This is a Rust reimplementation of Clang's main type-discrimination test:
// https://github.com/llvm/llvm-project/blob/main/clang/test/CodeGen/ptrauth-function-type-discriminator.c
// Variable and function names match the original C test.

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::Option::{None, Some};
use minicore::hint::black_box;
use minicore::mem::transmute;
use minicore::{Option, c_void, ptr};

extern "C" fn f() {}
extern "C" fn f2(_: i32) {}

// 1
// Rust function pointers are no-nullable, so this can not be expressed directly.
// ```c
//   void (*test_constant_null)(int) = 0;
// ```
// Use Option<TestConstantNullTy> instead.
type TestConstantNullTy = unsafe extern "C" fn(i32);

#[used]
// CHECK-DAG: @{{.*}}TEST_CONSTANT_NULL = constant {{.*}} zeroinitializer,
static TEST_CONSTANT_NULL: Option<TestConstantNullTy> = None;
#[used]
// DISC-DAG: @{{.*}}TEST_CONSTANT_NON_NULL = constant ptr ptrauth (ptr @{{.*}}f2, i32 0, i64 2712), align 8
// NO_DISC-DAG: @{{.*}}TEST_CONSTANT_NON_NULL = constant ptr ptrauth (ptr @{{.*}}f2, i32 0), align 8
static TEST_CONSTANT_NON_NULL: Option<TestConstantNullTy> = Some(f2);

// 2
// Clang expects to generate the discriminator based on the "casted to" type
// ```c
//   void f(void);
//   void (*test_constant_cast)(int) = (void (*)(int))f;
// ```
// Rust does not allow incompatible function pointer casts. `transmute` seems to be the closes to
// the cast.
#[used]
// DISC-DAG: @{{.*}}TEST_CONSTANT_CAST = constant ptr ptrauth (ptr @{{.*}}f, i32 0, i64 2712), align 8
// NO_DISC-DAG: @{{.*}}TEST_CONSTANT_CAST = constant ptr ptrauth (ptr @{{.*}}f, i32 0), align 8
static TEST_CONSTANT_CAST: unsafe extern "C" fn(i32) = unsafe { transmute(f as extern "C" fn()) };

// 3
// Clang can handle incomplete enum declaration, collapsing it to int:
// ```c
//   enum Enum0;
//   void enum_func(enum Enum0);
//   void (*enum_func_ptr)(enum Enum0) = enum_func;
// ```
// Mimic it with type Enum0 assigned to i32 and `__opaque`.
type Enum0 = i32;
#[repr(C)]
enum Enum1 {
    __opaque,
}
extern "C" {
    fn enum_func(arg: Enum0);
}
unsafe extern "C" fn enum_func_1(_x: Enum1) {}
#[used]
// DISC-DAG: @{{.*}}TEST_ENUM_FUNC_PTR = constant ptr ptrauth (ptr @{{.*}}enum_func, i32 0, i64 2712), align 8
// NO_DISC-DAG: @{{.*}}TEST_ENUM_FUNC_PTR = constant ptr ptrauth (ptr @{{.*}}enum_func, i32 0), align 8
static TEST_ENUM_FUNC_PTR: unsafe extern "C" fn(Enum0) = enum_func;
#[used]
// DISC-DAG: @{{.*}}TEST_ENUM_FUNC_PTR_1 = constant ptr ptrauth (ptr @{{.*}}enum_func_1, i32 0, i64 2712), align 8
// NO_DISC-DAG: @{{.*}}TEST_ENUM_FUNC_PTR_1 = constant ptr ptrauth (ptr @{{.*}}enum_func_1, i32 0), align 8
static TEST_ENUM_FUNC_PTR_1: unsafe extern "C" fn(Enum1) = enum_func_1;

// 4
// Rust can't fn -> *mut c_void casts. Use a chain of transmute.
// ```c
//   void *test_opaque =
//   #ifdef __cplusplus
//       (void *)
//   #endif
//       (void (*)(int))(double (*)(double))f;
// ```
// We expect zero-discriminator.
#[used]
// CHECK-DAG: @{{.*}}TEST_OPAQUE = {{.*}} ptr ptrauth (ptr @{{.*}}f, i32 0), align 8
static mut TEST_OPAQUE: *const c_void = unsafe {
    let p: extern "C" fn(f64) -> f64 = transmute::<extern "C" fn(), extern "C" fn(f64) -> f64>(f);
    transmute::<extern "C" fn(f64) -> f64, *const c_void>(p)
};
#[used]
// Also test a case that uses: as *const c_void.
// CHECK-DAG: @{{.*}}TEST_OPAQUE_1 = {{.*}} ptr ptrauth (ptr @{{.*}}f, i32 0), align 8
static mut TEST_OPAQUE_1: *const c_void = f as *const () as *const c_void;

// 5
// ```c
//   unsigned long test_intptr_t = (unsigned long)f;
// ```
// This is explicitly forbidden in Rust. Hypothetically we could get it through:
// #[used]
// static TEST_INTPTR_T: usize = f as usize;
// #[used]
// static TEST_INTPTR_T_1: usize = unsafe {
//     transmute::<extern "C" fn(), usize>(f)
// };
// But the compiler would not allow that issuing an error:
// error: pointers cannot be cast to integers during const eval
// And diagnostic:
// * for TEST_INTPTR_T:
// | static TEST_INTPTR_T: usize = f as usize;
// |                               ^^^^^^^^^^
// |
// = note: at compile-time, pointers do not have an integer value
// = note: avoiding this restriction via `transmute`, `union`, or raw pointers leads to compile-time undefined behavior
// * for TEST_INTPTR_T_1:
// | static TEST_INTPTR_T_1: usize = unsafe {
// | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ evaluation of `TEST_INTPTR_T_1` failed here
// |
// = help: this code performed an operation that depends on the underlying bytes representing a pointer
// = help: the absolute address of a pointer is not known at compile-time, so such operations are not supported
//
// The same limitation applies to:
// 6
// ```c
//   void (*test_through_long)(int) = (void (*)(int))(long)f;
// ```
// 7
// ```c
//   long test_to_long = (long)(double (*)())f;
// ```

extern "C" fn external_function() {}

// 8 and 9
// In Rust function automatically decays to function pointer. Furthermore, `&` used on function is
// not meant to produce a pointer to the function, instead it generates a reference to the function
// item. Use an intermediate `REF` variable to perform a round trip through reference.
// ```c
//   void (*fptr1)(void) = external_function;
//   void (*fptr2)(void) = &external_function;
// ```
#[used]
// DISC-DAG: @{{.*}}FPTR1 = constant ptr ptrauth (ptr @{{.*}}external_function, i32 0, i64 18983), align 8
// NO_DISC-DAG: @{{.*}}FPTR1 = constant ptr ptrauth (ptr @{{.*}}external_function, i32 0), align 8
static FPTR1: extern "C" fn() = external_function;
// 9
#[used]
static REF: &extern "C" fn() = &(external_function as extern "C" fn());
#[used]
// DISC-DAG: @{{.*}}FPTR2 = constant ptr ptrauth (ptr @{{.*}}external_function, i32 0, i64 18983), align 8
// NO_DISC-DAG: @{{.*}}FPTR2 = constant ptr ptrauth (ptr @{{.*}}external_function, i32 0), align 8
static FPTR2: extern "C" fn() = *REF;

// Rust doesn't support `__builtin_ptrauth_blend_discriminator` or `__builtin_ptrauth_sign_constant`
// builtins.
// 10
// ```c
//   void (*fptr3)(void) = __builtin_ptrauth_sign_constant(&external_function, 2, 26);
// ```
// 11
// ```c
//   void (*fptr4)(void) = __builtin_ptrauth_sign_constant(&external_function, 2, __builtin_ptrauth_blend_discriminator(&fptr4, 26));
// ```

// 12
// Test calling through a global function pointer.
// ```c
//   void (*fnptr)(void);
//   void test_call() {
//     fnptr();
//   }
// ```
#[used]
// DISC-DAG: @{{.*}}FNPTR = {{.*}}ptr ptrauth (ptr @{{.*}}external_function, i32 0, i64 18983), align 8
// NO_DISC-DAG: @{{.*}}FNPTR = {{.*}}ptr ptrauth (ptr @{{.*}}external_function, i32 0), align 8
static mut FNPTR: extern "C" fn() = external_function;
// CHECK-LABEL {{.*}}test_call
pub unsafe fn test_call() {
    // CHECK: [[FNPTR_PTR:%.*]] = load ptr, ptr @{{.*}}FNPTR, align 8
    // DISC: call void [[FNPTR_PTR]]() {{.*}} "ptrauth"(i32 0, i64 18983) ]
    // NO_DISC: call void [[FNPTR_PTR]]() {{.*}} "ptrauth"(i32 0, i64 0) ]
    FNPTR();
}

// 13
// ```c
//   void (*test_function_pointer())(void) {
//     return external_function;
//   }
// ```
// CHECK-LABEL: @{{.*}}test_function_pointer
pub extern "C" fn test_function_pointer() -> extern "C" fn() {
    // DISC:   ret ptr ptrauth (ptr @{{.*}}external_function, i32 0, i64 18983)
    // NO_DISC:   ret ptr ptrauth (ptr @{{.*}}external_function, i32 0)
    external_function
}

// 14
// C tests that the discriminator is stable when a struct type transitions from incomplete to
// complete. Rust has no notion of type completion, so this case has no direct equivalent.
// ```c
//   struct InitiallyIncomplete;
//   extern struct InitiallyIncomplete returns_initially_incomplete(void);
//
//   void use_while_incomplete() {
//     struct InitiallyIncomplete (*fnptr)(void) = &returns_initially_incomplete;
//   }
//
//   struct InitiallyIncomplete { int x; };
//   void use_while_complete() {
//     struct InitiallyIncomplete (*fnptr)(void) = &returns_initially_incomplete;
//   }
// ```
// Test each case in isolation (complete/incomplete) - the difference in discrimnators is expected.
#[repr(C)]
pub struct InitiallyIncomplete {
    _private: [u8; 0],
}

extern "C" fn returns_initially_incomplete() -> InitiallyIncomplete {
    InitiallyIncomplete { _private: [] }
}

// CHECK-LABEL: @{{.*}}use_while_incomplete
pub unsafe fn use_while_incomplete() {
    // DISC: call ptr @{{.*}}InitiallyIncomplete{{.*}}(ptr ptrauth (ptr @{{.*}}returns_initially_incomplete, i32 0, i64 25106))
    // NO_DISC: call ptr @{{.*}}InitiallyIncomplete{{.*}}(ptr ptrauth (ptr @{{.*}}returns_initially_incomplete, i32 0))
    let INITIALLY_INCOMPLETE_FNPTR: extern "C" fn() -> InitiallyIncomplete =
        returns_initially_incomplete;

    black_box(INITIALLY_INCOMPLETE_FNPTR);
}

#[repr(C)]
pub struct InitiallyComplete {
    x: i32,
}
extern "C" fn returns_initially_complete() -> InitiallyComplete {
    { InitiallyComplete { x: 42 } }
}
// CHECK-LABEL: @{{.*}}use_while_complete
pub fn use_while_complete() {
    // DISC: call ptr @{{.*}}InitiallyComplete{{.*}}(ptr ptrauth (ptr @{{.*}}returns_initially_complete, i32 0, i64 9528))
    // NO_DISC: call ptr @{{.*}}InitiallyComplete{{.*}}(ptr ptrauth (ptr @{{.*}}returns_initially_complete, i32 0))
    let INITIALLY_COMPLETE_FNPTR: extern "C" fn() -> InitiallyComplete = returns_initially_complete;
    black_box(INITIALLY_COMPLETE_FNPTR);
}

// 15
// K&R function definition can be expressed in Rust and in any case a function definition without a
// prototype is deprecated in all versions of C and is not supported in C23
// ```c
//   void knr(param)
//     int param;
//   {}
//
//   void test_knr() {
//     void (*p)() = knr;
//     p(0);
//   }
// ```

// 16
// Rust does not allow for redeclaration of functions
// ```c
//   void test_redeclaration() {
//     void redecl();
//     void (*ptr)() = redecl;
//     void redecl(int);
//     void (*ptr2)(int) = redecl;
//     ptr();
//     ptr2(0);
//   }
// ```

// 17
// This is redeclaration of functions using Kernighan and Ritchie notation, not supported.
// ```c
//   void knr2(param)
//        int param;
// {}
//
// void test_redecl_knr() {
//   void (*p)() = knr2;
//   p();
//
//   void knr2(int);
//
//   void (*p2)(int) = knr2;
//   p2(0);
//
// }
// ```
