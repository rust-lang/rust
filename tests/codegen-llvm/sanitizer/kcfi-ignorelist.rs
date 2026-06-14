//@ needs-sanitizer-kcfi
//@ compile-flags: -Zsanitizer=kcfi -C panic=abort -Cunsafe-allow-abi-mismatch=sanitizer -Zsanitizer-ignorelist={{src-base}}/sanitizer/ignorelist.txt

#![crate_type = "lib"]

// CHECK: define void @test_kcfi
// CHECK-NOT: !kcfi_type
#[no_mangle]
pub fn test_kcfi(x: &mut i32) {
    *x = 1;
}

// CHECK: define void @test_memory
// CHECK-SAME: !kcfi_type
#[no_mangle]
pub fn test_memory(x: &mut i32) {
    *x = 2;
}
