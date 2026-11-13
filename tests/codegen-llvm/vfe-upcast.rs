//@ compile-flags: -Zvirtual-function-elimination -Clto=fat -Copt-level=3 -Zverify-llvm-ir
//@ no-prefer-dynamic
#![crate_type = "staticlib"]

// Check that a vtable also gets a !type annotation for any supertraits it has whose vtable is a
// prefix of the base trait's. For `trait Foo: A + B` this is true for A because it is the first,
// and also for B if it has no trait methods.
//
// Without this annotation, -Zvirtual-function-elimination may incorrectly remove a vtable if it
// only used by the supertrait.

use std::hint::black_box;

pub trait Super {
    fn s(&self) -> i32;
}

pub trait Sub: Super {}

pub struct A;
pub struct B;

impl Super for A {
    fn s(&self) -> i32 {
        black_box(1)
    }
}

impl Sub for A {}

impl Super for B {
    fn s(&self) -> i32 {
        black_box(2)
    }
}

// CHECK-LABEL: @call_super
#[unsafe(no_mangle)]
pub fn call_super(x: &dyn Super) -> i32 {
    // This call should not be devirtualized, x might be either A or B.
    // CHECK: call noundef i32 %{{.*}}(ptr
    x.s()
}

#[unsafe(no_mangle)]
pub fn call_a() -> i32 {
    let a: &dyn Sub = black_box(&A);
    call_super(a)
}

#[unsafe(no_mangle)]
pub fn call_b() -> i32 {
    call_super(black_box(&B))
}
