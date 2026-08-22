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
// Make sure that rustc can correctly handle resigning Option<T>, this makes an interesting use
// case, as options are the only way of expressing nullable pointer
//
// Equivalent C:
// #include <stdbool.h>
// #include <stddef.h>
//
// typedef void (*src_fn_t)(int);
// typedef void (*dst_fn_t)(void);
//
// void f(void) {}
// void g(int x) { (void)x; }
//
// struct SrcOpt {
//   src_fn_t value;
// };
//
// struct DstOpt {
//   dst_fn_t value;
// };
//
// struct SrcOptRef {
//   src_fn_t *value;
// };
//
// struct DstOptRef {
//   dst_fn_t *value;
// };
//
// struct W1 {
//   dst_fn_t value;
// };
//
// static src_fn_t F = g;
//
// void test_option(void) {
//   dst_fn_t p = (dst_fn_t)g;
//
//   if (p)
//     p();
// }
//
// dst_fn_t test_option_none(void) { return NULL; }
//
// dst_fn_t test_option_runtime_none(bool x) {
//   src_fn_t p = x ? g : NULL;
//
//   return (dst_fn_t)p;
// }
//
// dst_fn_t test_option_runtime(src_fn_t p) { return (dst_fn_t)p; }
//
// struct DstOpt test_transparent_wrapper_option_none(void) {
//   struct SrcOpt src = {NULL};
//
//   return (struct DstOpt){(dst_fn_t)src.value};
// }
//
// struct DstOpt test_transparent_wrapper_option_runtime(src_fn_t p) {
//   struct SrcOpt src = {p};
//
//   return (struct DstOpt){(dst_fn_t)src.value};
// }
//
// struct DstOpt test_transparent_wrapper_option_branch(bool x) {
//   struct SrcOpt src = {x ? g : NULL};
//
//   return (struct DstOpt){(dst_fn_t)src.value};
// }
//
// struct DstOptRef test_transparent_wrapper_option_static_none(void) {
//   return (struct DstOptRef){NULL};
// }
//
// struct DstOptRef test_transparent_wrapper_option_static_some(void) {
//   return (struct DstOptRef){(dst_fn_t *)&F};
// }
//
// struct W1 test_option_inside_wrapper(src_fn_t p) {
//   return (struct W1){(dst_fn_t)p};
// }

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]

extern crate minicore;

use minicore::Option;
use minicore::Option::{None, Some};
use minicore::mem::transmute;

// DISC-DAG: @{{.*}}F = internal constant ptr ptrauth (ptr @{{.*}}g, i32 0, i64 2712)
static F: extern "C" fn(i32) = g;

extern "C" fn f() {}
extern "C" fn g(_: i32) {}

// CHECK-LABEL-DAG: test_option
pub fn test_option() {
    unsafe {
        // DISC: [[PTR:%.*]] = icmp eq i64 ptrtoint (ptr ptrauth (ptr @{{.*}}g, i32 0, i64 2712) to i64), 0
        // DISC: br i1 [[PTR]], label %ptrauth.null, label %ptrauth.resign

        // DISC: ptrauth.null:
        // DISC: store ptr null, ptr %{{.*}}
        // DISC: br label %ptrauth.end

        // DISC: ptrauth.resign:
        // DISC: [[RESIGNED:%.*]] = call i64 @llvm.ptrauth.resign(i64 ptrtoint (ptr ptrauth (ptr {{.*}}g, i32 0, i64 2712) to i64), i32 0, i64 2712, i32 0, i64 18983)
        // DISC: [[RESIGNED_PTR:%.*]] = inttoptr i64 [[RESIGNED]] to ptr
        // DISC: store ptr [[RESIGNED_PTR]], ptr %{{.*}}

        // NO_DISC: store ptr ptrauth (ptr @{{.*}}g, i32 0), ptr %{{.*}}
        let p: Option<extern "C" fn()> = transmute(Some(g as extern "C" fn(i32)));

        // DISC: bb1:
        // DISC: [[FP:%.*]] = load ptr, ptr %{{.*}}
        // DISC: call void [[FP]]() #{{.*}} [ "ptrauth"(i32 0, i64 18983) ]
        if let Some(fp) = p {
            // NO_DIS: call void %{{>*}}() #[[#]] [ "ptrauth"(i32 0, i64 0) ]
            fp();
        }
    }
}

// CHECK-LABEL-DAG: test_option_none
pub fn test_option_none() -> Option<extern "C" fn()> {
    unsafe {
        // CHECK: ret ptr null
        let p: Option<extern "C" fn()> = transmute::<Option<extern "C" fn(i32)>, _>(None);
        p
    }
}

pub fn test_option_runtime_none(x: bool) -> Option<extern "C" fn()> {
    // The source Option<fn(i32)> construction.
    // CHECK: bb2:
    // CHECK: store ptr null, ptr %{{.*}}

    // CHECK: bb1:
    // DISC: store ptr ptrauth (ptr @{{.*}}g, i32 0, i64 2712), ptr %{{.*}}
    // NO_DISC: store ptr ptrauth (ptr @{{.*}}g, i32 0), ptr [[RES:%.*]]

    // The nullable transmute checks for None before resigning.
    // DISC: [[PTR:%.*]] = icmp eq i64 %{{.*}}, 0
    // DISC: br i1 [[PTR]], label %ptrauth.null, label %ptrauth.resign

    // None path.
    // DISC: ptrauth.null:
    // DISC: store ptr null, ptr %{{.*}}
    // DISC: br label %ptrauth.end

    // Some path.
    // DISC: ptrauth.resign:
    // DISC: [[RESIGNED:%.*]] = call i64 @llvm.ptrauth.resign(i64 %{{.*}}, i32 0, i64 2712, i32 0, i64 18983)
    // DISC: [[RESIGNED_PTR:%.*]] = inttoptr i64 [[RESIGNED]] to ptr
    // DISC: store ptr [[RESIGNED_PTR]], ptr %{{.*}}

    // Result.
    // DISC: ptrauth.end:
    // DISC: ret ptr %{{.*}}
    // NO_DISC: [[RETURN_VAL:%.*]] = load ptr, ptr [[RES]]
    // NO_DISC: ret ptr [[RETURN_VAL]]
    unsafe {
        let p = if x { Some(g as extern "C" fn(i32)) } else { None };

        transmute(p)
    }
}

// CHECK-LABEL-DAG: test_option_runtime
pub unsafe fn test_option_runtime(p: Option<extern "C" fn(i32)>) -> Option<extern "C" fn()> {
    // Check nullable discrimination.
    // DISC: [[PTR:%.*]] = ptrtoint ptr %p to i64
    // DISC: [[IS_NULL:%.*]] = icmp eq i64 [[PTR]], 0
    // DISC: br i1 [[IS_NULL]], label %ptrauth.null, label %ptrauth.resign

    // Null path.
    // DISC: ptrauth.null:
    // DISC: store ptr null, ptr %{{.*}}
    // DISC: br label %ptrauth.end

    // Non-null is resigned from fn(i32) to fn().
    // DISC: ptrauth.resign:
    // DISC: [[RESIGNED:%.*]] = call i64 @llvm.ptrauth.resign(i64 %{{.*}}, i32 0, i64 2712, i32 0, i64 18983)
    // DISC: [[RESIGNED_PTR:%.*]] = inttoptr i64 [[RESIGNED]] to ptr
    // DISC: store ptr [[RESIGNED_PTR]], ptr %{{.*}}

    // Return.
    // DISC: ptrauth.end:
    // DISC: [[RET:%.*]] = load ptr, ptr %{{.*}}
    // DISC: ret ptr [[RET]]
    // NO_DISC: ret ptr %{{.*}}p
    transmute(p)
}

#[repr(transparent)]
struct SrcOpt(Option<extern "C" fn(i32)>);

#[repr(transparent)]
struct DstOpt(Option<extern "C" fn()>);

// CHECK-LABEL-DAG: test_transparent_wrapper_option_none
pub fn test_transparent_wrapper_option_none() -> DstOpt {
    // CHECK: ret ptr null
    unsafe { transmute(SrcOpt(None)) }
}

// CHECK-LABEL-DAG: test_transparent_wrapper_option_runtime
pub fn test_transparent_wrapper_option_runtime(p: Option<extern "C" fn(i32)>) -> DstOpt {
    // Transparent wrapper should still reach the nullable fn pointer path.
    // DISC: [[PTR:%.*]] = ptrtoint ptr %p to i64
    // DISC: [[IS_NULL:%.*]] = icmp eq i64 [[PTR]], 0
    // DISC: br i1 [[IS_NULL]], label %ptrauth.null, label %ptrauth.resign

    // DISC: ptrauth.null:
    // DISC: store ptr null, ptr %{{.*}}
    // DISC: br label %ptrauth.end

    // DISC: ptrauth.resign:
    // DISC: [[RESIGNED:%.*]] = call i64 @llvm.ptrauth.resign(i64 %{{.*}}, i32 0, i64 2712, i32 0, i64 18983)
    // DISC: [[RESIGNED_PTR:%.*]] = inttoptr i64 [[RESIGNED]] to ptr
    // DISC: store ptr [[RESIGNED_PTR]], ptr %{{.*}}

    // DISC: ptrauth.end:
    // DISC: ret ptr %{{.*}}

    // NO_DISC: ret ptr %{{.*}}p
    unsafe { transmute(SrcOpt(p)) }
}

// CHECK-LABEL-DAG: test_transparent_wrapper_option_branch
pub fn test_transparent_wrapper_option_branch(x: bool) -> DstOpt {
    // Source Option construction branching.
    // DISC: br i1 %x, label %bb1, label %bb2

    // None arm.
    // DISC: bb2:
    // DISC: store ptr null, ptr %{{.*}}

    // Some arm.
    // DISC: bb1:
    // DISC: store ptr ptrauth (ptr @{{.*}}g, i32 0, i64 2712), ptr %{{.*}}

    // Transparent wrapper must still produce nullable resign.
    // DISC: [[IS_NULL:%.*]] = icmp eq i64 %{{.*}}, 0
    // DISC: br i1 [[IS_NULL]], label %ptrauth.null, label %ptrauth.resign

    // DISC: ptrauth.resign:
    // DISC: call i64 @llvm.ptrauth.resign(i64 %{{.*}}, i64 2712, i32 0, i64 18983)
    let p = if x { Some(g as extern "C" fn(i32)) } else { None };

    unsafe { transmute(SrcOpt(p)) }
}

#[repr(transparent)]
struct SrcOptRef(Option<&'static extern "C" fn(i32)>);

#[repr(transparent)]
struct DstOptRef(Option<&'static extern "C" fn()>);

// CHECK-LABEL-DAG: test_transparent_wrapper_option_static_none
pub fn test_transparent_wrapper_option_static_none() -> DstOptRef {
    // DISC: ret ptr null
    unsafe { transmute(SrcOptRef(None)) }
}

// CHECK-LABEL-DAG: test_transparent_wrapper_option_static_some
pub fn test_transparent_wrapper_option_static_some() -> DstOptRef {
    // The pointer being returned is a reference to a function pointer object, not the function
    // pointer value itself.
    // DISC-NOT: llvm.ptrauth.resign
    // DISC: ret ptr @{{.*}}F
    unsafe { transmute(SrcOptRef(Some(&F))) }
}

#[repr(transparent)]
struct W1<T>(T);

// CHECK-LABEL-DAG: test_option_inside_wrapper
pub fn test_option_inside_wrapper(p: Option<extern "C" fn(i32)>) -> W1<Option<extern "C" fn()>> {
    // DISC: [[NULL_CHECK:%.*]] = ptrtoint ptr %p to i64
    // DISC: [[IS_NULL:%.*]] = icmp eq i64 [[NULL_CHECK]], 0
    // DISC: br i1 [[IS_NULL]], label %ptrauth.null, label %ptrauth.resign

    // DISC: ptrauth.null:
    // DISC: store ptr null, ptr %{{.*}}
    // DISC: br label %ptrauth.end

    // DISC: ptrauth.resign:
    // DISC: [[RESIGN_PTR:%.*]] = ptrtoint ptr %p to i64
    // DISC: [[RESIGNED:%.*]] = call i64 @llvm.ptrauth.resign(i64 [[RESIGN_PTR]], i32 0, i64 2712, i32 0, i64 18983)

    // DISC: ptrauth.end:
    // DISC: ret ptr
    unsafe { transmute(W1(p)) }
}
