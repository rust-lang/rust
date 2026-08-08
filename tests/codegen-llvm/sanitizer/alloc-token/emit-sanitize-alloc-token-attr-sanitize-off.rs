// Verifies that the `#[sanitize(alloc_token = "off")]` attribute can be used to selectively
// disable the `sanitize_alloc_token` function attribute.
//
//@ needs-sanitizer-alloc-token
//@ min-llvm-version: 22
//@ compile-flags: -Cno-prepopulate-passes -Copt-level=0 -Ctarget-feature=-crt-static -Cunsafe-allow-abi-mismatch=sanitizer -Zsanitizer=alloc-token

#![crate_name = "alloc_token_sanitize_off"]
#![crate_type = "lib"]
#![feature(sanitize)]

// CHECK-LABEL: alloc_token_sanitize_off::unsanitized
// CHECK:       Function Attrs: {{.*}}
// CHECK-NOT:   sanitize_alloc_token
// CHECK:       start:
#[sanitize(alloc_token = "off")]
pub fn unsanitized() {}

// CHECK-LABEL: alloc_token_sanitize_off::sanitized
// CHECK:       Function Attrs: {{.*}}sanitize_alloc_token{{.*}}
// CHECK:       start:
pub fn sanitized() {}
