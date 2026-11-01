//@ needs-sanitizer-cfi
//@ compile-flags: -Zsanitizer=cfi -Clto -Ctarget-feature=-crt-static -Cunsafe-allow-abi-mismatch=sanitizer -Zsanitizer-ignorelist={{src-base}}/sanitizer/ignorelist/type-ignorelist.txt

#![crate_type = "lib"]

// CHECK: define void @test_type
// CHECK-SAME: !type
// CHECK-NOT: trap
// CHECK: call void %f()
// CHECK: trap
#[no_mangle]
pub fn test_type(f: fn(), g: fn(i32), x: &mut i32) {
    *x = 1;
    f();
    g(1);
}

// CHECK: define void @test_type_2()
// CHECK-SAME: !type
#[no_mangle]
pub fn test_type_2() {}
