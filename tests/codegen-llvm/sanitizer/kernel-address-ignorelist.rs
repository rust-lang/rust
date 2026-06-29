//@ needs-sanitizer-address
//@ compile-flags: -Zsanitizer=address -Zsanitizer-ignorelist={{src-base}}/sanitizer/kernel-address-ignorelist.txt

#![crate_type = "lib"]

// CHECK: ; Function Attrs:
// CHECK-SAME: sanitize_address
// CHECK-NEXT: define void @test_kernel_address_ignored
#[no_mangle]
pub fn test_kernel_address_ignored(x: &mut i32) {
    *x = 1;
}

// CHECK: ; Function Attrs:
// CHECK-NOT: sanitize_address
// CHECK-NEXT: define void @test_address_ignored
#[no_mangle]
pub fn test_address_ignored(x: &mut i32) {
    *x = 2;
}
