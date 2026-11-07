//@ compile-flags: -Zvirtual-function-elimination -Clto=fat -Copt-level=3 -Zverify-llvm-ir
//@ no-prefer-dynamic
//@ min-llvm-version: 23
#![crate_type = "staticlib"]

// Check that vtables include a !type annotation for supertraits whose vtable are a prefix.
//
// What is still missing is that for any supertrait whose vtable is not a prefix, its vtable does
// not get any !type annotation. That means we provide incomplete information to VFE and it may
// incorrectly replace a virtual call with a direct call to the only callee VFE knows about.

use std::hint::black_box;

// FIXME: C_VTABLE should have a !type annotation for C, but it currently has none.
// CHECK: @vtable.{{[0-9]+}} = {{.*}}3Foo3foo{{.*}} !type ![[META_SIZED:[0-9]+]], !type ![[A:[0-9]+]], !type ![[B:[0-9]+]], !type ![[FOO:[0-9]+]], !guid
// CHECK: @[[C_VTABLE:alloc_[0-9a-f]+]] = {{.*}}1C1c }>, align {{[0-9]+}}, !guid
// CHECK: @vtable.{{[0-9]+}} = {{.*}}1C1c, ptr @[[C_VTABLE]], ptr {{.*}}3Bar3bar{{.*}} !type ![[META_SIZED]], !type ![[A]], !type ![[BAR:[0-9]+]], !guid

// CHECK-DAG: ![[META_SIZED]] = !{i64 0, !"{{.*}}9MetaSized"}
// CHECK-DAG: ![[A]] = !{i64 0, !"{{.*}}1A"}
// CHECK-DAG: ![[B]] = !{i64 0, !"{{.*}}1B"}
// CHECK-DAG: ![[FOO]] = !{i64 0, !"{{.*}}3Foo"}
// CHECK-DAG: ![[BAR]] = !{i64 0, !"{{.*}}3Bar"}

pub trait A {
    fn a(&self) -> i32;
}

pub trait B {}

pub trait C {
    fn c(&self) -> i32;
}

/// A trait with two supertraits:
///
/// - A is a prefix of Foo, because it is the first supertrait in the list
/// - B is a prefix of Foo, because it has no methods
pub trait Foo: A + B {
    fn foo(&self) -> i32;
}

/// A trait with two supertraits:
///
/// - A is a prefix of Bar, because it is the first supertrait in the list
/// - C is a NOT a prefix of Bar, because it has methods
pub trait Bar: A + C {
    fn bar(&self) -> i32;
}

// The remainder is uninteresting, and only here so that the optimizer can't optimize out the
// interesting bits.

pub struct S;

impl A for S {
    fn a(&self) -> i32 {
        black_box(1)
    }
}

impl B for S {}

impl C for S {
    fn c(&self) -> i32 {
        black_box(2)
    }
}

impl Foo for S {
    fn foo(&self) -> i32 {
        black_box(3)
    }
}

impl Bar for S {
    fn bar(&self) -> i32 {
        black_box(4)
    }
}

#[unsafe(no_mangle)]
pub fn use_foo(x: &dyn Foo) -> i32 {
    x.foo() + x.a()
}

#[unsafe(no_mangle)]
pub fn use_bar(x: &dyn Bar) -> i32 {
    x.bar() + x.a() + x.c()
}

#[unsafe(no_mangle)]
pub fn use_s() -> i32 {
    use_foo(black_box(&S)) + use_bar(black_box(&S))
}
