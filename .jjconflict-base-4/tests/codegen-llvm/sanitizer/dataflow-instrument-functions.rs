// Verifies that functions are instrumented.
//
//@ needs-sanitizer-dataflow
//@ compile-flags: -Copt-level=0 -Zsanitizer=dataflow -C unsafe-allow-abi-mismatch=sanitizer

#![crate_type = "lib"]

pub fn foo() {}
// CHECK: define{{.*}}foo{{.*}}.dfsan
