//! This test checks that we do not monomorphize functions that are only
//! used to evaluate static items, but never used in runtime code.

//@compile-flags: --crate-type=lib -Copt-level=0

#![feature(generic_const_items)]

const fn foo() {}

pub static FOO: () = foo();

// CHECK-NOT: define{{.*}}foo{{.*}}

const fn bar() {}

pub const BAR: () = bar();

// CHECK-NOT: define{{.*}}bar{{.*}}

const fn baz() {}

#[rustfmt::skip]
pub const BAZ<const C: bool>: () = if C {
    baz()
};

// CHECK: define{{.*}}baz{{.*}}
