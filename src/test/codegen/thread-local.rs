// compile-flags: -O
// aux-build:thread_local_aux.rs
// ignore-windows FIXME(#84933)
// ignore-wasm globals are used instead of thread locals
// ignore-emscripten globals are used instead of thread locals
// ignore-android does not use #[thread_local]

#![crate_type = "lib"]

extern crate thread_local_aux as aux;

use std::cell::Cell;

thread_local!(static A: Cell<u32> = const { Cell::new(1) });

// CHECK: [[TLS_AUX:@.+]] = external thread_local local_unnamed_addr global i64
// CHECK: [[TLS:@.+]] = internal thread_local unnamed_addr global

// CHECK-LABEL: @get
#[no_mangle]
fn get() -> u32 {
    // CHECK: %0 = load i32, i32* {{.*}}[[TLS]]{{.*}}
    // CHECK-NEXT: ret i32 %0
    A.with(|a| a.get())
}

// CHECK-LABEL: @set
#[no_mangle]
fn set(v: u32) {
    // CHECK: store i32 %0, i32* {{.*}}[[TLS]]{{.*}}
    // CHECK-NEXT: ret void
    A.with(|a| a.set(v))
}

// CHECK-LABEL: @get_aux
#[no_mangle]
fn get_aux() -> u64 {
    // CHECK: %0 = load i64, i64* [[TLS_AUX]]
    // CHECK-NEXT: ret i64 %0
    aux::A.with(|a| a.get())
}

// CHECK-LABEL: @set_aux
#[no_mangle]
fn set_aux(v: u64) {
    // CHECK: store i64 %0, i64* [[TLS_AUX]]
    // CHECK-NEXT: ret void
    aux::A.with(|a| a.set(v))
}
