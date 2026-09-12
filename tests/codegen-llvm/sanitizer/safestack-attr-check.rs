// This tests that the safestack attribute is applied when enabling the safe-stack sanitizer.
//
//@ needs-sanitizer-safestack
//@ compile-flags: -Zsanitizer=safestack -Copt-level=0 -C unsafe-allow-abi-mismatch=sanitizer

#![crate_type = "lib"]
#![feature(sanitize)]

// CHECK: ; Function Attrs:{{.*}}safestack
// CHECK-NEXT: define void @tagged
#[no_mangle]
pub fn tagged() {}

// CHECK: ; Function Attrs:
// CHECK-NOT: safestack
// CHECK: define void @untagged
#[no_mangle]
#[sanitize(safestack = "off")]
pub fn untagged() {}
