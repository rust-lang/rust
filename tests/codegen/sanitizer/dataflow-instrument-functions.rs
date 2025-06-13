// Verifies that functions are instrumented.
//
//@ needs-sanitizer-dataflow
//@ compile-flags: -Copt-level=0 -Zsanitizer=dataflow
//@ compile-flags: -C unsafe-allow-abi-mismatch=sanitizer

#![crate_type = "lib"]

pub fn foo() {}
// CHECK: define{{.*}}foo{{.*}}.dfsan
