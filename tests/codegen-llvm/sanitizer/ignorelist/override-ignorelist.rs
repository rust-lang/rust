//@ needs-sanitizer-address
//@ compile-flags: -Zsanitizer=address -Ctarget-feature=-crt-static -Zsanitizer-ignorelist={{src-base}}/sanitizer/ignorelist/override-ignorelist.txt

#![crate_type = "lib"]

// CHECK: ; Function Attrs:
// CHECK-NOT: sanitize_address
// CHECK-NEXT: define void @test_ignored
#[no_mangle]
pub fn test_ignored(x: &mut i32) {
    *x = 1;
}

// CHECK: ; Function Attrs:
// CHECK-SAME: sanitize_address
// CHECK-NEXT: define void @test_re_enabled
#[no_mangle]
pub fn test_re_enabled(x: &mut i32) {
    *x = 2;
}

pub static RE_ENABLED_REF: fn(&mut i32) = test_mangled_re_enabled;

// CHECK: ; Function Attrs:
// CHECK-SAME: sanitize_address
// CHECK-LABEL: define {{.*}}test_mangled_re_enabled
#[inline(never)]
pub fn test_mangled_re_enabled(x: &mut i32) {
    *x = 3;
}
