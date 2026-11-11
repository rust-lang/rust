//@ needs-sanitizer-kasan
//@ compile-flags: -Zsanitizer=kernel-address -Ctarget-feature=-crt-static -Zsanitizer-ignorelist={{src-base}}/sanitizer/ignorelist/override-kernel-address.txt

#![crate_type = "lib"]

// [address] does not ignore functions under -Zsanitizer=kernel-address (matches Clang):
// CHECK: ; Function Attrs:
// CHECK-SAME: sanitize_address
// CHECK-NEXT: define void @test_address_not_ignored
#[no_mangle]
pub fn test_address_not_ignored(x: &mut i32) {
    *x = 1;
}

// Ignored via [kernel-address]:
// CHECK: ; Function Attrs:
// CHECK-NOT: sanitize_address
// CHECK-NEXT: define void @test_kernel_ignored
#[no_mangle]
pub fn test_kernel_ignored(x: &mut i32) {
    *x = 2;
}

// Re-enabled via [kernel-address] =sanitize:
// CHECK: ; Function Attrs:
// CHECK-SAME: sanitize_address
// CHECK-NEXT: define void @test_kernel_override
#[no_mangle]
pub fn test_kernel_override(x: &mut i32) {
    *x = 3;
}
