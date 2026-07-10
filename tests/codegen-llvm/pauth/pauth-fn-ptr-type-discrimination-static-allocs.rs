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
// Test pointer authentication generation in compile-time constants. Covers standalone function
// pointer constants, promoted temporaries, immutable and mutable statics, arrays of function
// pointers, and mixed structs containing function pointers with different signatures and
// discriminators.

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]
extern crate minicore;
use minicore::Sync;

// DISC-DAG: @[[SCALAR_ALLOC:alloc_[0-9a-f]+]] = private unnamed_addr constant ptr ptrauth (ptr {{.*}}foo{{.*}}, i32 0, i64 18983)
// NO_DISC-DAG: @[[SCALAR_ALLOC:alloc_[0-9a-f]+]] = private unnamed_addr constant ptr ptrauth (ptr {{.*}}foo{{.*}}, i32 0)

// DISC-DAG: @[[MIXED_ALLOC:alloc_[0-9a-f]+]] = private unnamed_addr constant <{ ptr, ptr, ptr, ptr }> <{ ptr ptrauth (ptr @{{.*}}foo, i32 0, i64 18983), ptr ptrauth (ptr @{{.*}}foo_i32, i32 0, i64 2712), ptr ptrauth (ptr @{{.*}}foo_i64, i32 0, i64 2712), ptr ptrauth (ptr @{{.*}}foo_ret, i32 0, i64 42271) }>
// NO_DISC-DAG: @[[MIXED_ALLOC:alloc_[0-9a-f]+]] = private unnamed_addr constant <{ ptr, ptr, ptr, ptr }> <{ ptr ptrauth (ptr @{{.*}}foo, i32 0), ptr ptrauth (ptr @{{.*}}foo_i32, i32 0), ptr ptrauth (ptr @{{.*}}foo_i64, i32 0), ptr ptrauth (ptr @{{.*}}foo_ret, i32 0) }>

// DISC-DAG: @[[STATIC_MIXED:.*STATIC_MIXED]] = constant <{ ptr, ptr, ptr, ptr }> <{ ptr ptrauth (ptr @{{.*}}foo, i32 0, i64 18983), ptr ptrauth (ptr @{{.*}}foo_i32, i32 0, i64 2712), ptr ptrauth (ptr @{{.*}}foo_i64, i32 0, i64 2712), ptr ptrauth (ptr @{{.*}}foo_ret, i32 0, i64 42271) }>
// NO_DISC-DAG: @[[STATIC_MIXED:.*STATIC_MIXED]] = constant <{ ptr, ptr, ptr, ptr }> <{ ptr ptrauth (ptr @{{.*}}foo, i32 0), ptr ptrauth (ptr @{{.*}}foo_i32, i32 0), ptr ptrauth (ptr @{{.*}}foo_i64, i32 0), ptr ptrauth (ptr @{{.*}}foo_ret, i32 0) }>

// DISC-DAG: @[[STATIC_TABLE:.*STATIC_TABLE]] = constant <{ ptr, ptr }> <{ ptr ptrauth (ptr @{{.*}}foo, i32 0, i64 18983), ptr ptrauth (ptr @{{.*}}bar, i32 0, i64 18983) }>
// NO_DISC-DAG: @[[STATIC_TABLE:.*STATIC_TABLE]] = constant <{ ptr, ptr }> <{ ptr ptrauth (ptr @{{.*}}foo, i32 0), ptr ptrauth (ptr @{{.*}}bar, i32 0) }>

// DISC-DAG: @[[MUT_TABLE:.*MUT_TABLE]] = global <{ ptr, ptr }> <{ ptr ptrauth (ptr @{{.*}}foo, i32 0, i64 18983), ptr ptrauth (ptr @{{.*}}bar, i32 0, i64 18983) }>
// NO_DISC-DAG: @[[MUT_TABLE:.*MUT_TABLE]] = global <{ ptr, ptr }> <{ ptr ptrauth (ptr @{{.*}}foo, i32 0), ptr ptrauth (ptr @{{.*}}bar, i32 0) }>

extern "C" fn foo() {}
extern "C" fn foo_i32(_: i32) {}
extern "C" fn foo_i64(_: i64) {}
extern "C" fn foo_ret() -> i32 {
    0
}
extern "C" fn bar() {}

const F: extern "C" fn() = foo;

// CHECK-LABEL: test_scalar
pub fn test_scalar() {
    // DISC: call void ptrauth (ptr @{{.*}}foo{{.*}}, i32 0, i64 18983)()
    // NO_DISC: call void ptrauth (ptr @{{.*}}foo{{.*}}, i32 0)()
    let p: &'static extern "C" fn() = &F;
    p();
}

#[repr(C)]
struct Mixed {
    a: extern "C" fn(),
    b: extern "C" fn(i32),
    c: extern "C" fn(i64),
    d: extern "C" fn() -> i32,
}

impl Sync for Mixed {}

// CHECK-LABEL: test_promoted_mixed
pub fn test_promoted_mixed() {
    // CHECK: %{{.*}} = load ptr, ptr @[[MIXED_ALLOC]]
    // DISC: call void %{{.*}}() #[[#]] [ "ptrauth"(i32 0, i64 18983) ]
    // NO_DISC: call void %{{.*}}() #[[#]] [ "ptrauth"(i32 0, i64 0) ]
    // CHECK: %{{.*}} = load ptr, ptr getelementptr inbounds (i8, ptr @[[MIXED_ALLOC]], i64 8),
    // DISC: call void %{{.*}}(i32 1) #[[#]] [ "ptrauth"(i32 0, i64 2712) ]
    // NO_DISC: call void %{{.*}}(i32 1) #[[#]] [ "ptrauth"(i32 0, i64 0) ]
    // CHECK: %{{.*}} = load ptr, ptr getelementptr inbounds (i8, ptr @[[MIXED_ALLOC]], i64 16),
    // DISC: call void %{{.*}}(i64 1) #[[#]] [ "ptrauth"(i32 0, i64 2712) ]
    // NO_DISC: call void %{{.*}}(i64 1) #[[#]] [ "ptrauth"(i32 0, i64 0) ]
    // CHECK: %{{.*}} = load ptr, ptr getelementptr inbounds (i8, ptr @[[MIXED_ALLOC]], i64 24),
    // DISC: call i32 %{{.*}}() #[[#]] [ "ptrauth"(i32 0, i64 42271) ]
    // NO_DISC: call i32 %{{.*}}() #[[#]] [ "ptrauth"(i32 0, i64 0) ]
    let x: &'static Mixed = &Mixed { a: foo, b: foo_i32, c: foo_i64, d: foo_ret };

    (x.a)();
    (x.b)(1);
    (x.c)(1);
    let _ = (x.d)();
}

#[used]
static STATIC_MIXED: Mixed = Mixed { a: foo, b: foo_i32, c: foo_i64, d: foo_ret };

// CHECK-LABEL: test_static_mixed
pub fn test_static_mixed() {
    // CHECK: %{{.*}} = load ptr, ptr @[[STATIC_MIXED]]
    // DISC: call void %{{.*}}() #[[#]] [ "ptrauth"(i32 0, i64 18983) ]
    // NO_DISC: call void %{{.*}}() #[[#]] [ "ptrauth"(i32 0, i64 0) ]
    // CHECK: %{{.*}} = load ptr, ptr getelementptr inbounds (i8, ptr @[[STATIC_MIXED]], i64 8),
    // DISC: call void %{{.*}}(i32 1) #[[#]] [ "ptrauth"(i32 0, i64 2712) ]
    // NO_DISC: call void %{{.*}}(i32 1) #[[#]] [ "ptrauth"(i32 0, i64 0) ]
    (STATIC_MIXED.a)();
    (STATIC_MIXED.b)(1);
}

#[used]
static STATIC_TABLE: [extern "C" fn(); 2] = [foo, bar];

// CHECK-LABEL: test_static_array
pub fn test_static_array() {
    let p = &raw const STATIC_TABLE as *const extern "C" fn();

    // CHECK: call ptr {{.*}}read_volatile{{.*}}(ptr @[[STATIC_TABLE]])
    // DISC: call void %{{.*}}() #[[#]] [ "ptrauth"(i32 0, i64 18983) ]
    // NO_DISC: call void %{{.*}}() #[[#]] [ "ptrauth"(i32 0, i64 0) ]
    unsafe {
        (minicore::ptr::read_volatile(p))();
    }
}

#[used]
static mut MUT_TABLE: [extern "C" fn(); 2] = [foo, bar];

// CHECK-LABEL: test_mut_static
pub unsafe fn test_mut_static() {
    let p = &raw const MUT_TABLE as *const extern "C" fn();

    // CHECK: call ptr {{.*}}read_volatile{{.*}}(ptr @[[MUT_TABLE]])
    // DISC: call void %{{.*}}() #[[#]] [ "ptrauth"(i32 0, i64 18983) ]
    // NO_DISC: call void %{{.*}}() #[[#]] [ "ptrauth"(i32 0, i64 0) ]
    (minicore::ptr::read_volatile(p))();
}

// C equivalent:
// #include <stdint.h>
//
// void foo(void) {}
// void foo_i32(int32_t x) { (void)x; }
// void foo_i64(int64_t x) { (void)x; }
// int32_t foo_ret(void) { return 0; }
// void bar(void) {}
//
// void (*const F)(void) = foo;
//
// void test_scalar(void) {
//   void (*const *p)(void) = &F;
//   (*p)();
// }
//
// struct Mixed {
//   void (*a)(void);
//   void (*b)(int32_t);
//   void (*c)(int64_t);
//   int32_t (*d)(void);
// };
//
// void test_promoted_mixed(void) {
//   static const struct Mixed x = {
//       .a = foo,
//       .b = foo_i32,
//       .c = foo_i64,
//       .d = foo_ret,
//   };
//
//   x.a();
//   x.b(1);
//   x.c(1);
//   (void)x.d();
// }
//
// __attribute__((used)) const struct Mixed STATIC_MIXED = {
//     .a = foo,
//     .b = foo_i32,
//     .c = foo_i64,
//     .d = foo_ret,
// };
//
// void test_static_mixed(void) {
//   STATIC_MIXED.a();
//   STATIC_MIXED.b(1);
// }
//
// __attribute__((used)) void (*const STATIC_TABLE[2])(void) = {
//     foo,
//     bar,
// };
//
// void test_static_array(void) {
//   STATIC_TABLE[0]();
// }
//
// void (*MUT_TABLE[2])(void) = {
//     foo,
//     bar,
// };
//
// void test_mut_static(void) { MUT_TABLE[0](); }
