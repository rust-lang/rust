//@ add-minicore
//@ needs-llvm-components: aarch64
//@ compile-flags: --target=arm64e-apple-ios -Copt-level=0 -Cpanic=abort

#![feature(no_core, unboxed_closures)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

// CHECK-DAG: @RUST_POINTER = {{.*}}ptr ptrauth (ptr @rust_target, i32 0)
// CHECK-DAG: @RUST_TABLE = {{.*}}ptr ptrauth (ptr @rust_target, i32 0), ptr ptrauth (ptr @rust_other, i32 0)
// CHECK-DAG: @[[VTABLE:vtable\.[0-9]+]] = {{.*}}ptr ptrauth (ptr @[[GET:[^, )]+]], i32 0)
#[no_mangle]
pub static RUST_POINTER: fn() -> usize = rust_target;

#[no_mangle]
pub static RUST_TABLE: [fn() -> usize; 2] = [rust_target, rust_other];

pub trait Getter {
    fn get(&self) -> usize;
}

impl Getter for usize {
    fn get(&self) -> usize {
        *self
    }
}

// CHECK-LABEL: define {{.*}} @call_dyn_fn(
// CHECK: call i64 %{{.*}}({{.*}}){{.*}}[ "ptrauth"(i32 0, i64 0) ]
#[no_mangle]
pub fn call_dyn_fn(f: &dyn Fn() -> usize) -> usize {
    f()
}

// CHECK-LABEL: define {{.*}} @call_dyn_trait(
// CHECK: call i64 %{{.*}}({{.*}}){{.*}}[ "ptrauth"(i32 0, i64 0) ]
#[no_mangle]
pub fn call_dyn_trait(f: &dyn Getter) -> usize {
    f.get()
}

// CHECK-LABEL: define {{.*}} @call_rust(
// CHECK: call i64 %{{.*}}(){{.*}}[ "ptrauth"(i32 0, i64 0) ]
#[no_mangle]
pub fn call_rust(f: fn() -> usize) -> usize {
    f()
}

// CHECK-LABEL: define {{.*}} @make_dyn_trait(
// CHECK: insertvalue { ptr, ptr } {{.*}}, ptr @[[VTABLE]], 1
#[no_mangle]
pub fn make_dyn_trait(value: &usize) -> &dyn Getter {
    value
}

#[no_mangle]
pub fn rust_other() -> usize {
    9
}

#[no_mangle]
pub fn rust_target() -> usize {
    7
}

// CHECK-LABEL: define {{.*}} @take_closure(
// CHECK: ret ptr ptrauth (ptr @{{[^, )]+}}, i32 0)
#[no_mangle]
pub fn take_closure() -> fn() -> usize {
    || 11
}

// CHECK-LABEL: define {{.*}} @take_rust(
// CHECK: ret ptr ptrauth (ptr @rust_target, i32 0)
#[no_mangle]
pub fn take_rust() -> fn() -> usize {
    rust_target
}
