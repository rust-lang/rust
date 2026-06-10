//@ needs-sanitizer-kcfi
//@ compile-flags: -Zsanitizer=kcfi -Cpanic=abort -Cunsafe-allow-abi-mismatch=sanitizer -Zsanitizer-ignorelist={{src-base}}/sanitizer/ignorelist/type-ignorelist-kcfi.txt

#![crate_type = "lib"]

// CHECK: define void @test_type
// CHECK-SAME: !kcfi_type
// CHECK-NOT: [ "kcfi"
// CHECK: call void %f()
// CHECK: call void %g(i32 {{.*}}1){{.*}}[ "kcfi"
#[no_mangle]
pub fn test_type(f: fn(), g: fn(i32), x: &mut i32) {
    *x = 1;
    f();
    g(1);
}

// CHECK: define void @test_type_2()
// CHECK-SAME: !kcfi_type
#[no_mangle]
pub fn test_type_2() {}
