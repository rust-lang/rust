//@ add-minicore
//@ needs-llvm-components: aarch64
//@ revisions: ARM64E PAUTHTEST
//@ compile-flags: -Copt-level=0 -Cpanic=abort
//@ [ARM64E] compile-flags: --target=arm64e-apple-ios
//@ [PAUTHTEST] compile-flags: --target=aarch64-unknown-linux-pauthtest

#![feature(no_core)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;

extern "C" {
    fn callback();
    fn other_callback();
    static DATA: usize;
}

type Callback = unsafe extern "C" fn();
const CALLBACK: Callback = callback;
const CALLBACK_PAIR: (Callback, Callback) = (callback, other_callback);

// CHECK: @STATIC_CALLBACK = {{.*}}ptr ptrauth (ptr @callback, i32 0)
#[no_mangle]
pub static STATIC_CALLBACK: Callback = callback;

// CHECK-LABEL: define {{.*}} @const_as_raw(
// CHECK: ret ptr ptrauth (ptr @callback, i32 0)
#[no_mangle]
pub fn const_as_raw() -> *const () {
    CALLBACK as *const ()
}

// CHECK-LABEL: define {{.*}} @get_const(
// CHECK: ret ptr ptrauth (ptr @callback, i32 0)
#[no_mangle]
pub fn get_const() -> Callback {
    CALLBACK
}

// CHECK-LABEL: define {{.*}} @get_data(
// CHECK: ret ptr @DATA
#[no_mangle]
pub unsafe fn get_data() -> *const usize {
    &raw const DATA
}

// CHECK-LABEL: define {{.*}} @get_pair(
// CHECK: ret { ptr, ptr } { ptr ptrauth (ptr @callback, i32 0), ptr ptrauth (ptr @other_callback, i32 0) }
#[no_mangle]
pub fn get_pair() -> (Callback, Callback) {
    CALLBACK_PAIR
}

// CHECK-LABEL: define {{.*}} @get_reified(
// CHECK: ret ptr ptrauth (ptr @callback, i32 0)
#[no_mangle]
pub fn get_reified() -> Callback {
    callback
}
