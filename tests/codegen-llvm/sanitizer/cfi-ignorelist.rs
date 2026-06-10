//@ needs-sanitizer-cfi
//@ compile-flags: -Zsanitizer=cfi -Clto -Cunsafe-allow-abi-mismatch=sanitizer -Zsanitizer-ignorelist={{src-base}}/sanitizer/ignorelist.txt

#![crate_type = "lib"]

// CHECK: define void @test_cfi
// CHECK-NOT: !type
#[no_mangle]
pub fn test_cfi(f: fn(), x: &mut i32) {
    *x = 1;
    // CHECK-NOT: trap
    f();
}

// CHECK: define void @test_memory
// CHECK-SAME: !type
#[no_mangle]
pub fn test_memory(f: fn(i32), x: &mut i32) {
    *x = 2;
    // CHECK: trap
    f(1);
}
