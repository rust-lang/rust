// Verifies that the `sanitize_alloc_token` function attribute is emitted.
//
//@ needs-sanitizer-alloc-token
//@ min-llvm-version: 22
//@ compile-flags: -Cno-prepopulate-passes -Copt-level=0 -Ctarget-feature=-crt-static -Cunsafe-allow-abi-mismatch=sanitizer -Zsanitizer=alloc-token

#![crate_type = "lib"]

// CHECK-LABEL: emit_sanitize_alloc_token_attr::foo
// CHECK:       Function Attrs: {{.*}}sanitize_alloc_token{{.*}}
pub fn foo() {}
