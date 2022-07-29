// Verifies that "CFI Canonical Jump Tables" module flag is added.
//
// ignore-windows
// needs-sanitizer-cfi
// only-aarch64
// only-x86_64
// compile-flags: -Clto -Zsanitizer=cfi

#![crate_type="lib"]

pub fn foo() {
}

// CHECK: !{{[0-9]+}} = !{i32 2, !"CFI Canonical Jump Tables", i32 1}
