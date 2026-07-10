// ignore-tidy-file-linelength
//@ add-minicore
//@ only-pauthtest
// Run it at O0, so that the compiler doesn't optimise the calls away.

//@ revisions: DISC NO_DISC
//@ [DISC] needs-llvm-components: aarch64
//@ [DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=+function-pointer-type-discrimination -C opt-level=0

// Test generation of function-pointer type discriminators. The discriminator values were obtained
// from Clang by compiling equivalent C code (included). Both compilers must generate identical
// values.
//
// Tests function-pointer type discriminator generation for pointer authentication across nested
// static allocations, wrapper references, and padded structs.

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]
extern crate minicore;
use minicore::Sync;
use minicore::hint::black_box;

extern "C" fn foo(_: f32) {}
extern "C" fn bar(_: i32) {}

impl Sync for InnerA {}
impl Sync for InnerB {}
impl Sync for Outer {}

// DISC-DAG: @[[INNER_A:[^ ]*T_INNER_A]] = constant ptr ptrauth (ptr @{{.*}}foo, i32 0, i64 21613)
// DISC-DAG: @[[INNER_B:[^ ]*T_INNER_B]] = constant ptr ptrauth (ptr @{{.*}}bar, i32 0, i64 2712)
// NO_DISC-DAG: @[[INNER_A:[^ ]*T_INNER_A]] = constant ptr ptrauth (ptr @{{.*}}foo, i32 0)
// NO_DISC-DAG: @[[INNER_B:[^ ]*T_INNER_B]] = constant ptr ptrauth (ptr @{{.*}}bar, i32 0)
// CHECK-DAG: @[[T_OUTER:[^ ]*T_OUTER]] = constant ptr @[[ALLOC:alloc_[0-9a-f]+]]
// CHECK-DAG: @[[ALLOC]] = private unnamed_addr constant <{ ptr, ptr }> <{ ptr @[[INNER_A]], ptr @[[INNER_B]] }>

// DISC-DAG: @[[HAS_FN_PTR:[^ ]*T_HAS_FN_PTR]] = constant ptr ptrauth (ptr @{{.*}}foo, i32 0, i64 21613)
// NO_DISC-DAG: @[[HAS_FN_PTR:[^ ]*T_HAS_FN_PTR]] = constant ptr ptrauth (ptr @{{.*}}foo, i32 0)
// CHECK-DAG: @[[WRAPPER_1:[^ ]*T_WRAPPER_1]] = constant <{ [8 x i8], ptr }> <{ [8 x i8] {{.*}}, ptr @[[HAS_FN_PTR]] }>
// CHECK-DAG: @[[WRAPPER_2_ALLOC:alloc_[0-9a-f]+]] = private unnamed_addr constant <{ [4 x i8], [4 x i8], ptr }> <{ [4 x i8] {{.*}}, [4 x i8] {{.*}}, ptr @[[WRAPPER_1]] }>
// CHECK-DAG: @[[WRAPPER_2:[^ ]*T_WRAPPER_2]] = constant ptr @[[WRAPPER_2_ALLOC]]

// DISC-DAG: @[[PADDED_INNER:[^ ]*T_PADDED_INNER]] = constant <{ [2 x i8], [6 x i8], ptr, [1 x i8], [7 x i8], ptr }> <{ [2 x i8] {{.*}}, [6 x i8] {{.*}}, ptr ptrauth (ptr @{{.*}}foo, i32 0, i64 21613), [1 x i8] {{.*}}, [7 x i8] {{.*}}, ptr ptrauth (ptr @{{.*}}bar, i32 0, i64 2712) }>
// NO_DISC-DAG: @[[PADDED_INNER:[^ ]*T_PADDED_INNER]] = constant <{ [2 x i8], [6 x i8], ptr, [1 x i8], [7 x i8], ptr }> <{ [2 x i8] {{.*}}, [6 x i8] {{.*}}, ptr ptrauth (ptr @{{.*}}foo, i32 0), [1 x i8] {{.*}}, [7 x i8] {{.*}}, ptr ptrauth (ptr @{{.*}}bar, i32 0) }>
// CHECK-DAG: @[[PADDED_OUTER:[^ ]*T_PADDED_OUTER]] = constant ptr @[[PADDED_ALLOC:alloc_[0-9a-f]+]]
// CHECK-DAG: @[[PADDED_ALLOC]] = private unnamed_addr constant <{ [1 x i8], [7 x i8], ptr, [4 x i8], [4 x i8] }> <{ [1 x i8] {{.*}}, [7 x i8] {{.*}}, ptr @[[PADDED_INNER]], [4 x i8] {{.*}}, [4 x i8] {{.*}} }>

#[repr(C)]
struct InnerA {
    f: extern "C" fn(f32),
}

#[repr(C)]
struct InnerB {
    g: extern "C" fn(i32),
}

#[repr(C)]
struct Outer {
    a: &'static InnerA,
    b: &'static InnerB,
}

#[used]
static T_INNER_A: InnerA = InnerA { f: foo };

#[used]
static T_INNER_B: InnerB = InnerB { g: bar };

#[used]
static T_OUTER: &Outer = &Outer { a: &T_INNER_A, b: &T_INNER_B };

// CHECK-LABEL: test_1_two_inners
pub fn test_1_two_inners() {
    // DISC: call void {{.*}}(float {{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 21613) ]
    // NO_DISC: call void {{.*}}(float {{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 0) ]
    black_box((T_INNER_A.f))(0.12f32);

    // DISC: call void {{.*}}(i32 {{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 2712) ]
    // NO_DISC: call void {{.*}}(i32 {{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 0) ]
    black_box((T_INNER_B.g))(22);

    // DISC: call void {{.*}}(float {{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 21613) ]
    // NO_DISC: call void {{.*}}(float {{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 0) ]
    black_box((T_OUTER.a.f))(0.32f32);

    // DISC: call void {{.*}}(i32 {{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 2712) ]
    // NO_DISC: call void {{.*}}(i32 {{.*}}) #{{.*}} [ "ptrauth"(i32 0, i64 0) ]
    black_box((T_OUTER.b.g))(42);
}

impl Sync for HasFnPtr {}
impl Sync for Wrapper1 {}
impl Sync for Wrapper2 {}

#[repr(C)]
struct HasFnPtr {
    f: extern "C" fn(_: f32),
}

#[repr(C)]
struct Wrapper1 {
    pad: u64,
    p: &'static HasFnPtr,
}

#[repr(C)]
struct Wrapper2 {
    pad: u32,
    w: &'static Wrapper1,
}

#[used]
static T_HAS_FN_PTR: HasFnPtr = HasFnPtr { f: foo };

#[used]
static T_WRAPPER_1: Wrapper1 = Wrapper1 { pad: 42, p: &T_HAS_FN_PTR };

#[used]
static T_WRAPPER_2: &'static Wrapper2 = &Wrapper2 { pad: 7, w: &T_WRAPPER_1 };

pub fn test_2_nested_wrappers() {
    // DISC: call void {{.*}}(float {{.*}}){{.*}}"ptrauth"(i32 0, i64 21613)
    // NO_DISC: call void {{.*}}(float {{.*}}){{.*}}"ptrauth"(i32 0, i64 0)
    black_box((T_HAS_FN_PTR.f))(0.22f32);
    // DISC: call void {{.*}}(float {{.*}}){{.*}}"ptrauth"(i32 0, i64 21613)
    // NO_DISC: call void {{.*}}(float {{.*}}){{.*}}"ptrauth"(i32 0, i64 0)
    black_box((T_WRAPPER_1.p.f))(0.32f32);
    // DISC: call void {{.*}}(float {{.*}}){{.*}}"ptrauth"(i32 0, i64 21613)
    // NO_DISC: call void {{.*}}(float {{.*}}){{.*}}"ptrauth"(i32 0, i64 0)
    black_box((T_WRAPPER_2.w.p.f))(0.42f32);
}

impl Sync for PaddedInner {}
impl Sync for PaddedOuter {}

#[repr(C)]
struct PaddedInner {
    pad: u16,
    f: extern "C" fn(f32),
    pad_2: u8,
    f_2: extern "C" fn(i32),
}

#[repr(C)]
struct PaddedOuter {
    pad_3: u8,
    inner: &'static PaddedInner,
    pad_4: u32,
}

#[used]
static T_PADDED_INNER: PaddedInner = PaddedInner { pad: 0, f: foo, pad_2: 1, f_2: bar };

#[used]
static T_PADDED_OUTER: &PaddedOuter = &PaddedOuter { pad_3: 2, pad_4: 3, inner: &T_PADDED_INNER };

// CHECK-LABEL: test_3_padded_structs
pub fn test_3_padded_structs() {
    // DISC: call void {{.*}}(float {{.*}}){{.*}}"ptrauth"(i32 0, i64 21613)
    // NO_DISC: call void {{.*}}(float {{.*}}){{.*}}"ptrauth"(i32 0, i64 0)
    black_box((T_PADDED_OUTER.inner.f))(0.32f32);
    // DISC: call void {{.*}}(i32 {{.*}}){{.*}}"ptrauth"(i32 0, i64 2712)
    // NO_DISC: call void {{.*}}(i32 {{.*}}){{.*}}"ptrauth"(i32 0, i64 0)
    black_box((T_PADDED_OUTER.inner.f_2))(42);
}

// C equivalent:
// #include <stdint.h>
//
// void foo(float _) {}
// void bar(int32_t _) {}
//
// struct InnerA {
//   void (*f)(float);
// };
//
// struct InnerB {
//   void (*g)(int32_t);
// };
//
// struct Outer {
//   const struct InnerA *a;
//   const struct InnerB *b;
// };
//
// static const struct InnerA T_INNER_A = {
//     .f = foo,
// };
//
// static const struct InnerB T_INNER_B = {
//     .g = bar,
// };
//
// static const struct Outer X_storage = {
//     .a = &T_INNER_A,
//     .b = &T_INNER_B,
// };
//
// static const struct Outer *const T_OUTER = &X_storage;
//
// int test_1_two_inners(void) {
//   T_INNER_A.f(.12f);
//   T_INNER_B.g(22);
//   T_OUTER->a->f(.32f);
//   T_OUTER->b->g(42);
//   return 0;
// }
//
// struct HasFnPtr {
//   void (*f)(float);
// };
//
// struct Wrapper1 {
//   uint64_t pad;
//   const struct HasFnPtr *p;
// };
//
// struct Wrapper2 {
//   uint32_t pad;
//   const struct Wrapper1 *w;
// };
//
// static const struct HasFnPtr T_HAS_FN_PTR = {
//     .f = foo,
// };
//
// static const struct Wrapper1 T_WRAPPER_1 = {
//     .pad = 42,
//     .p = &T_HAS_FN_PTR,
// };
//
// static const struct Wrapper2 T_WRAPPER_2_STORAGE = {
//     .pad = 7,
//     .w = &T_WRAPPER_1,
// };
//
// static const struct Wrapper2 *const T_WRAPPER_2 = &T_WRAPPER_2_STORAGE;
//
// void test_2_nested_wrappers(void) {
//   T_HAS_FN_PTR.f(0.22f);
//   T_WRAPPER_1.p->f(0.32f);
//   T_WRAPPER_2->w->p->f(0.42f);
// }
//
// struct PaddedInner {
//   uint16_t pad;
//   void (*f)(float);
//   uint8_t pad_2;
//   void (*f_2)(int32_t);
// };
//
// struct PaddedOuter {
//   uint8_t pad_3;
//   const struct PaddedInner *inner;
//   uint32_t pad_4;
// };
//
// __attribute__((used)) static const struct PaddedInner T_PADDED_INNER = {
//     .pad = 0,
//     .f = foo,
//     .pad_2 = 1,
//     .f_2 = bar,
// };
//
// __attribute__((used)) static const struct PaddedOuter T_PADDED_OUTER = {
//     .pad_3 = 2,
//     .inner = &T_PADDED_INNER,
//     .pad_4 = 3,
// };
//
// __attribute__((used)) static const struct PaddedOuter *const X =
//     &T_PADDED_OUTER;
//
// void test_3_padded_structs(void) {
//   X->inner->f(0.32f);
//   X->inner->f_2(42);
// }
