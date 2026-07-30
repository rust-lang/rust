// Verifies that "CFI Canonical Jump Tables" module flag is added.
//
//@ needs-sanitizer-cfi
//@ compile-flags: -Clto -Ctarget-feature=-crt-static -Tsanitizer=cfi -C unsafe-allow-abi-mismatch=sanitizer
//@ compile-flags: -Zunstable-options

#![crate_type = "lib"]

pub fn foo() {}

// CHECK: !{{[0-9]+}} = !{i32 4, !"CFI Canonical Jump Tables", i32 1}
