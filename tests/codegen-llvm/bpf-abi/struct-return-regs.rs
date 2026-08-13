//@ add-minicore
//@ revisions: bpfel bpfeb
//@[bpfel] compile-flags: --target=bpfel-unknown-none
//@[bpfeb] compile-flags: --target=bpfeb-unknown-none
//@ needs-llvm-components: bpf
//@ compile-flags: -Copt-level=3 -Cno-prepopulate-passes
//@ min-llvm-version: 23
#![feature(no_core)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

#[repr(C)]
struct Foo0 {
    a: i8,
}

#[repr(C)]
struct Foo1 {
    a: i32,
}

#[repr(C)]
struct Foo2 {
    a: i32,
    b: i64,
}

#[repr(C)]
struct Foo3 {
    a: i32,
    b: i32,
    c: i64,
}

impl Copy for Foo0 {}
impl Copy for Foo1 {}
impl Copy for Foo2 {}
impl Copy for Foo3 {}

// CHECK-LABEL: define{{.*}} i8 @bar0(
// CHECK: ret i8
#[no_mangle]
extern "C" fn bar0(a: i8) -> Foo0 {
    Foo0 { a }
}

// CHECK-LABEL: define{{.*}} i32 @bar1(
// CHECK: ret i32
#[no_mangle]
extern "C" fn bar1(a: i32) -> Foo1 {
    Foo1 { a }
}

// CHECK-LABEL: define{{.*}} [2 x i64] @bar2(
// CHECK: ret [2 x i64]
#[no_mangle]
extern "C" fn bar2(a: i32, b: i32) -> Foo2 {
    Foo2 { a, b: b as i64 }
}

// CHECK-LABEL: define{{.*}} [2 x i64] @bar3(
// CHECK: ret [2 x i64]
#[no_mangle]
extern "C" fn bar3(a: i32, b: i32, c: i32) -> Foo3 {
    Foo3 { a, b, c: c as i64 }
}

// CHECK-LABEL: define{{.*}} i8 @check0(
// CHECK: %[[C1:.*]] = call i8 @bar0(
// CHECK: store i8 %[[C1]]
#[no_mangle]
extern "C" fn check0(a: i8) -> i8 {
    let v = bar0(a);
    v.a
}

// CHECK-LABEL: define{{.*}} i32 @check1(
// CHECK: %[[C1:.*]] = call i32 @bar1(
// CHECK: store i32 %[[C1]]
#[no_mangle]
extern "C" fn check1(a: i32) -> i32 {
    let v = bar1(a);
    v.a
}

// CHECK-LABEL: define{{.*}} i32 @check2(
// CHECK: %[[C2:.*]] = call [2 x i64] @bar2(
// CHECK: store [2 x i64] %[[C2]]
#[no_mangle]
extern "C" fn check2(a: i32, b: i32) -> i32 {
    let v = bar2(a, b);
    hint::black_box(v);
    v.a
}

// CHECK-LABEL: define{{.*}} i32 @check3(
// CHECK: %[[C3:.*]] = call [2 x i64] @bar3(
// CHECK: store [2 x i64] %[[C3]]
#[no_mangle]
extern "C" fn check3(a: i32, b: i32, c: i32) -> i32 {
    let v = bar3(a, b, c);
    hint::black_box(v);
    v.a
}
