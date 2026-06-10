//@ needs-sanitizer-cfi
//@ compile-flags: -Zsanitizer=cfi -Clto -Cunsafe-allow-abi-mismatch=sanitizer -Zsanitizer-ignorelist={{src-base}}/sanitizer/ignorelist/cfi-ignorelist.txt

#![crate_type = "lib"]

// CHECK: define void @test_cfi
// CHECK-SAME: !type
// CHECK-NOT: trap
// CHECK: call void %f()
#[no_mangle]
pub fn test_cfi(f: fn(), x: &mut i32) {
    *x = 1;
    f();
}

// CHECK: define void @test_memory
// CHECK-SAME: !type
// CHECK: trap
#[no_mangle]
pub fn test_memory(f: fn(i32), x: &mut i32) {
    *x = 2;
    f(1);
}

// CHECK: define void @test_all
// CHECK-SAME: !type
// CHECK-NOT: trap
// CHECK: call void %f()
#[no_mangle]
pub fn test_all(f: fn(), x: &mut i32) {
    *x = 3;
    f();
}

// CHECK: define void @test_cfi_icall
// CHECK-SAME: !type
// CHECK-NOT: trap
// CHECK: call void %f()
#[no_mangle]
pub fn test_cfi_icall(f: fn(), x: &mut i32) {
    *x = 4;
    f();
}
