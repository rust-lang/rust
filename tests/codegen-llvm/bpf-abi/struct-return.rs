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
struct T1 {}

#[repr(C)]
struct T2 {
    a: i32,
}
impl Copy for T2 {}

#[repr(C)]
struct T3 {
    a: i32,
    b: i64,
}

#[repr(C)]
struct T4 {
    a: i64,
    b: i64,
    c: i64,
}

#[repr(C)]
struct T5 {
    a: i8,
}

#[repr(C)]
union U1 {
    a: i32,
    b: i64,
}

// CHECK: define{{.*}} void @foo1()
#[no_mangle]
extern "C" fn foo1() -> T1 {
    T1 {}
}

// CHECK: define{{.*}} i32 @foo2()
#[no_mangle]
extern "C" fn foo2() -> T2 {
    T2 { a: 0 }
}

// CHECK: define{{.*}} [2 x i64] @foo3()
#[no_mangle]
extern "C" fn foo3() -> T3 {
    T3 { a: 0, b: 0 }
}

// CHECK: define{{.*}} void @foo4(ptr{{.*}}sret([24 x i8]){{.*}}align 8
#[no_mangle]
extern "C" fn foo4() -> T4 {
    T4 { a: 0, b: 0, c: 0 }
}

// CHECK: define{{.*}} i8 @foo5()
#[no_mangle]
extern "C" fn foo5() -> T5 {
    T5 { a: 0 }
}

// CHECK: define{{.*}} i64 @foou()
#[no_mangle]
extern "C" fn foou() -> U1 {
    U1 { b: 0 }
}

// CHECK-LABEL: define{{.*}} i32 @bar()
// CHECK: %[[C2:.*]] = call i32 @foo2()
// CHECK: store i32 %[[C2]]
// CHECK: %[[C3:.*]] = call [2 x i64] @foo3()
// CHECK: store [2 x i64] %[[C3]]
#[no_mangle]
extern "C" fn bar() -> i32 {
    let a = foo2();
    let b = foo3();
    hint::black_box((a, b));
    a.a
}
