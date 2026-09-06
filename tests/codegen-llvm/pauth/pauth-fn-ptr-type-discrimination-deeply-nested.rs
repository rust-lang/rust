// ignore-tidy-file-linelength
//@ add-minicore
//@ only-pauthtest
// Run it at O0, so that the compiler doesn't optimise the calls away.
//@ revisions: DISC NO_DISC

//@ [DISC] needs-llvm-components: aarch64
//@ [DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=+function-pointer-type-discrimination -C opt-level=0
//@ [NO_DISC] needs-llvm-components: aarch64
//@ [NO_DISC] compile-flags: --target=aarch64-unknown-linux-pauthtest --crate-type=lib -Zpointer-authentication=-function-pointer-type-discrimination -C opt-level=0

// Make sure that the compiler can see through chains of nested structs, both when used as globals,
// arguments and returns.

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]
extern crate minicore;
use minicore::hint::black_box;

// Nested fn ptr chain.
type L0 = extern "C" fn(i32) -> i32;
type L1 = extern "C" fn(L0) -> i32;
type L2 = extern "C" fn(L1) -> i32;
type L3 = extern "C" fn(L2) -> i32;
type L4 = extern "C" fn(L3) -> i32;
// Function returning fn ptr.
type DeepRet = extern "C" fn() -> L0;

#[used]
// DISC: @{{.*}}T_DEEP = constant ptr ptrauth (ptr @{{.*}}f_deep, i32 0, i64 1059), align 8
// NO_DISC: @{{.*}}T_DEEP = constant ptr ptrauth (ptr @{{.*}}f_deep, i32 0), align 8
static T_DEEP: unsafe extern "C" fn(L4) -> DeepRet = f_deep;

// Leaf callback.
// CHECK-LABEL: callback_i32
pub extern "C" fn callback_i32(x: i32) -> i32 {
    x
}

// Dummy chain impl.
// CHECK-LABEL: dummy_l1
// CHECK: (ptr [[CB:%.*]])
pub extern "C" fn dummy_l1(cb: L0) -> i32 {
    // DISC: call i32 [[CB]](i32 5) {{.*}} [ "ptrauth"(i32 0, i64 2981) ]
    // NO_DISC: call i32 [[CB]](i32 5) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
    cb(5)
}
// CHECK-LABEL: dummy_l2
// CHECK: (ptr [[CB:%.*]])
pub extern "C" fn dummy_l2(cb: L1) -> i32 {
    // DISC: call i32 [[CB]](ptr ptrauth (ptr @{{.*}}callback_i32, i32 0, i64 2981)) {{.*}} [ "ptrauth"(i32 0, i64 12410) ]
    // NO_DISC: call i32 [[CB]](ptr ptrauth (ptr @{{.*}}callback_i32, i32 0)) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
    cb(callback_i32)
}
// CHECK-LABEL: dummy_l3
// CHECK: (ptr [[CB:%.*]])
pub extern "C" fn dummy_l3(cb: L2) -> i32 {
    // DISC: call i32 [[CB]](ptr ptrauth (ptr @{{.*}}dummy_l1, i32 0, i64 12410)) {{.*}} [ "ptrauth"(i32 0, i64 12410) ]
    // NO_DISC: call i32 [[CB]](ptr ptrauth (ptr @{{.*}}dummy_l1, i32 0)) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
    cb(dummy_l1)
}
// CHECK-LABEL: dummy_l4
// CHECK: (ptr [[CB:%.*]])
pub extern "C" fn dummy_l4(cb: L3) -> i32 {
    // DISC: call i32 [[CB]](ptr ptrauth (ptr @{{.*}}dummy_l2, i32 0, i64 12410)) {{.*}} [ "ptrauth"(i32 0, i64 12410) ]
    // NO_DISC: call i32 [[CB]](ptr ptrauth (ptr @{{.*}}dummy_l2, i32 0)) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
    cb(dummy_l2)
}
// Return fn impl.
// CHECK-LABEL: returned_fn
pub extern "C" fn returned_fn() -> L0 {
    // DISC: ret ptr ptrauth (ptr @{{.*}}callback_i32, i32 0, i64 2981)
    // NO_DISC: ret ptr ptrauth (ptr @{{.*}}callback_i32, i32 0)
    return callback_i32;
}
// Entry point to the chain, takes L4 and returns function returning fn ptr.
// CHECK-LABEL: f_deep
// CHECK: (ptr [[CB:%.*]])
pub extern "C" fn f_deep(cb: L4) -> DeepRet {
    // DISC: call i32 [[CB]](ptr ptrauth (ptr @{{.*}}dummy_l3, i32 0, i64 12410)) {{.*}} [ "ptrauth"(i32 0, i64 12410) ]
    // NO_DISC: call i32 [[CB]](ptr ptrauth (ptr @{{.*}}dummy_l3, i32 0)) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
    cb(dummy_l3);
    // DISC: ret ptr ptrauth (ptr @{{.*}}returned_fn, i32 0, i64 34128)
    // NO_DISC: ret ptr ptrauth (ptr @{{.*}}returned_fn, i32 0)
    return returned_fn;
}

// CHECK-LABEL: main
pub fn main() {
    unsafe {
        // DISC: [[RET_FN:%.*]] = call ptr ptrauth (ptr @{{.*}}f_deep, i32 0, i64 1059)(ptr ptrauth (ptr @{{.*}}dummy_l4, i32 0, i64 12410)) {{.*}} [ "ptrauth"(i32 0, i64 1059) ]
        // NO_DISC: [[RET_FN:%.*]] = call ptr ptrauth (ptr @{{.*}}f_deep, i32 0)(ptr ptrauth (ptr @{{.*}}dummy_l4, i32 0)) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let ret_fn: DeepRet = T_DEEP(dummy_l4);
        // DISC: [[INNER_FN:%.*]] = call ptr [[RET_FN]]() {{.*}} [ "ptrauth"(i32 0, i64 34128) ]
        // NO_DISC: [[INNER_FN:%.*]] = call ptr [[RET_FN]]() {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let inner_fn: L0 = ret_fn();
        // DISC: call i32 [[INNER_FN]](i32 42) {{.*}} [ "ptrauth"(i32 0, i64 2981) ]
        // NO_DISC: call i32 [[INNER_FN]](i32 42) {{.*}} [ "ptrauth"(i32 0, i64 0) ]
        let result = inner_fn(42);

        black_box(result);
    }
}
