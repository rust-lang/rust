//@ needs-sanitizer-cfi
//@ compile-flags: -Zsanitizer=cfi -Clto -Cunsafe-allow-abi-mismatch=sanitizer -Zsanitizer-ignorelist={{src-base}}/sanitizer/ignorelist/src-ignore.txt

#![crate_type = "lib"]

// CHECK: define void @test_file
// CHECK-SAME: !type
// CHECK-NOT: llvm.type.test
// CHECK-NOT: trap
// CHECK: call void %f()
#[no_mangle]
pub fn test_file(f: fn(), x: &mut i32) {
    *x = 1;
    f();
}
