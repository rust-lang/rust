//@ needs-sanitizer-cfi
//@ compile-flags: -Zsanitizer=cfi -Clto -Cunsafe-allow-abi-mismatch=sanitizer -Zsanitizer-ignorelist={{src-base}}/sanitizer/ignorelist.txt

#![crate_type = "lib"]

// CHECK: define void @test_type
// CHECK-SAME: !type
#[no_mangle]
pub fn test_type(f: fn(), x: &mut i32) {
    *x = 1;
    // CHECK-NOT: trap
    f();
}

// Ensure the function definition of test_type_2 has no !type metadata
// since it has the type `fn()` which is ignored
// CHECK: define void @test_type_2()
// CHECK-NOT: !type
#[no_mangle]
pub fn test_type_2() {}
