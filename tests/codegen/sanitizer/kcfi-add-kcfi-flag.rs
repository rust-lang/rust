// Verifies that "kcfi" module flag is added.
//
// needs-sanitizer-kcfi
// compile-flags: -Ctarget-feature=-crt-static -Zsanitizer=kcfi

#![crate_type="lib"]

pub fn foo() {
}

// CHECK: !{{[0-9]+}} = !{i32 4, !"kcfi", i32 1}
