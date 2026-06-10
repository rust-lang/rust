//@ needs-sanitizer-cfi
//@ compile-flags: -Zsanitizer=cfi -Clto -Cunsafe-allow-abi-mismatch=sanitizer -Zsanitizer-ignorelist={{src-base}}/sanitizer/ignorelist.txt

#![crate_type = "lib"]

// CHECK: define void @test_file
// CHECK-NOT: !type
#[no_mangle]
pub fn test_file(f: fn(), x: &mut i32) {
    *x = 1;
    // CHECK-NOT: trap
    f();
}
