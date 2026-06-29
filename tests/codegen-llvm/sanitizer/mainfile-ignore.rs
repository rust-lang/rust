//@ needs-sanitizer-address
//@ compile-flags: -Zsanitizer=address -Zsanitizer-ignorelist={{src-base}}/sanitizer/mainfile-ignorelist.txt

#![crate_type = "lib"]

// CHECK: ; Function Attrs:
// CHECK-NOT: sanitize_address
// CHECK-NEXT: define void @test_mainfile
#[no_mangle]
pub fn test_mainfile(x: &mut i32) {
    *x = 1;
}
