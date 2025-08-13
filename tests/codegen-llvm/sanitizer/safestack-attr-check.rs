// This tests that the safestack attribute is applied when enabling the safe-stack sanitizer.
//
//@ needs-sanitizer-safestack
//@ compile-flags: -Zsanitizer=safestack -Copt-level=0
//@ compile-flags: -C unsafe-allow-abi-mismatch=sanitizer

#![crate_type = "lib"]

// CHECK: ; Function Attrs:{{.*}}safestack
pub fn tagged() {}

// CHECK: attributes #0 = {{.*}}safestack
