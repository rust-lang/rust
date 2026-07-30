// This tests that the sanitize_memtag attribute is
// applied when enabling the memtag sanitizer.
//
//@ needs-sanitizer-memtag
//@ compile-flags: -C unsafe-allow-abi-mismatch=sanitizer -Tsanitizer=memtag -Ctarget-feature=+mte -Copt-level=0
//@ compile-flags: -Zunstable-options

#![crate_type = "lib"]

// CHECK: ; Function Attrs:{{.*}}sanitize_memtag
pub fn tagged() {}

// CHECK: attributes #0 = {{.*}}sanitize_memtag
