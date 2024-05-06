//! This test checks that we do not monomorphize functions that are only
//! used to evaluate static items, but never used in runtime code.

//@compile-flags: --crate-type=lib -Copt-level=0

const fn foo() {}

pub static FOO: () = foo();

// CHECK-NOT: define{{.*}}foo{{.*}}
