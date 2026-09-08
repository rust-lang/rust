//@ needs-sanitizer-kcfi
//@ compile-flags: -Zsanitizer=kcfi -C panic=abort -Ctarget-feature=-crt-static -Cunsafe-allow-abi-mismatch=sanitizer -Zsanitizer-ignorelist={{src-base}}/sanitizer/ignorelist/kcfi-ignorelist.txt

#![crate_type = "lib"]

// CHECK: define void @test_kcfi
// CHECK-SAME: !kcfi_type
// CHECK-NOT: [ "kcfi"
// CHECK: call void %f()
#[no_mangle]
pub fn test_kcfi(f: fn(), x: &mut i32) {
    *x = 1;
    f();
}

// CHECK: define void @test_memory
// CHECK-SAME: !kcfi_type
// CHECK: call void %f(i32 {{.*}}1){{.*}}[ "kcfi"
#[no_mangle]
pub fn test_memory(f: fn(i32), x: &mut i32) {
    *x = 2;
    f(1);
}
