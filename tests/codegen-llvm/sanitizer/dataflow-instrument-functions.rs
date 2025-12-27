// Verifies that functions are instrumented.
//
//@ needs-sanitizer-dataflow
//@ compile-flags: -Copt-level=0 -Cunsafe-allow-abi-mismatch=sanitize -Zunstable-options -Csanitize=dataflow

#![crate_type = "lib"]

pub fn foo() {}
// CHECK: define{{.*}}foo{{.*}}.dfsan
