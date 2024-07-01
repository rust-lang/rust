// Verifies that functions are instrumented.
//
//@ needs-sanitizer-dataflow
//@ compile-flags: -Copt-level=0 -Zunstable-options -Csanitizer=dataflow

#![crate_type = "lib"]

pub fn foo() {}
// CHECK: define{{.*}}foo{{.*}}.dfsan
