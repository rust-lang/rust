//@ compile-flags: -Copt-level=3
//@ aux-build:thread_local_aux.rs
//@ ignore-windows FIXME(#134939)
//@ ignore-wasm globals are used instead of thread locals
//@ ignore-emscripten globals are used instead of thread locals
//@ ignore-android does not use #[thread_local]
//@ ignore-nto does not use #[thread_local]

#![crate_type = "lib"]

extern crate thread_local_aux as aux;

use std::cell::Cell;

thread_local!(static A: Cell<u32> = const { Cell::new(1) });

// CHECK: [[TLS_AUX:@.+]] = external thread_local{{.*}} global i64
// CHECK: [[TLS:@.+]] = internal thread_local{{.*}} global

// CHECK-LABEL: @get
#[no_mangle]
fn get() -> u32 {
    // CHECK: [[PTR:%.+]] = tail call {{.*}} ptr @llvm.threadlocal.address.p0(ptr [[TLS]])
    // CHECK-NEXT: [[RET_0:%.+]] = load i32, ptr [[PTR]]
    // CHECK-NEXT: ret i32 [[RET_0]]
    A.with(|a| a.get())
}

// CHECK-LABEL: @set
#[no_mangle]
fn set(v: u32) {
    // CHECK: [[PTR:%.+]] = tail call {{.*}} ptr @llvm.threadlocal.address.p0(ptr [[TLS]])
    // CHECK-NEXT: store i32 %0, ptr [[PTR]]
    // CHECK-NEXT: ret void
    A.with(|a| a.set(v))
}

// CHECK-LABEL: @get_aux
#[no_mangle]
fn get_aux() -> u64 {
    // CHECK: [[PTR:%.+]] = tail call {{.*}} ptr @llvm.threadlocal.address.p0(ptr [[TLS_AUX]])
    // CHECK-NEXT: [[RET_1:%.+]] = load i64, ptr [[PTR]]
    // CHECK-NEXT: ret i64 [[RET_1]]
    aux::A.with(|a| a.get())
}

// CHECK-LABEL: @set_aux
#[no_mangle]
fn set_aux(v: u64) {
    // CHECK: [[PTR:%.+]] = tail call {{.*}} ptr @llvm.threadlocal.address.p0(ptr [[TLS_AUX]])
    // CHECK-NEXT: store i64 %0, ptr [[PTR]]
    // CHECK-NEXT: ret void
    aux::A.with(|a| a.set(v))
}
