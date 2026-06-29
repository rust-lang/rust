//@ needs-sanitizer-kcfi
//@ compile-flags: -Zsanitizer=kcfi -Cpanic=abort -Cunsafe-allow-abi-mismatch=sanitizer -Zsanitizer-ignorelist={{src-base}}/sanitizer/ignorelist.txt

#![crate_type = "lib"]

// CHECK: define void @test_type
// CHECK-SAME: !kcfi_type
#[no_mangle]
pub fn test_type(f: fn(), x: &mut i32) {
    *x = 1;
    // CHECK-NOT: !kcfi_type
    f();
}

// CHECK: define void @test_type_2()
// CHECK-NOT: !kcfi_type
#[no_mangle]
pub fn test_type_2() {}
