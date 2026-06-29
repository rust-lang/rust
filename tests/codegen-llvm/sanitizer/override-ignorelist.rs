//@ needs-sanitizer-address
//@ compile-flags: -Zsanitizer=address -Zsanitizer-ignorelist={{src-base}}/sanitizer/override-ignorelist.txt

#![crate_type = "lib"]

// CHECK: ; Function Attrs:
// CHECK-NOT: sanitize_address
// CHECK-NEXT: define void @test_ignored
#[no_mangle]
pub fn test_ignored(x: &mut i32) {
    *x = 1;
}

// CHECK: ; Function Attrs:
// CHECK-SAME: sanitize_address
// CHECK-NEXT: define void @test_re_enabled
#[no_mangle]
pub fn test_re_enabled(x: &mut i32) {
    *x = 2;
}
