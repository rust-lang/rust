// Verifies that "cfi-normalize-integers" module flag is added.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Ctarget-feature=-crt-static -Zsanitizer=cfi -Zsanitizer-cfi-normalize-integers -C unsafe-allow-abi-mismatch=sanitizer,sanitizer-cfi-normalize-integers

#![crate_type = "lib"]

pub fn foo() {}

// CHECK: !{{[0-9]+}} = !{i32 4, !"cfi-normalize-integers", i32 1}
