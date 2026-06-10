//@ needs-sanitizer-address
//@ compile-flags: -Zsanitizer=address -Zsanitizer-ignorelist={{src-base}}/sanitizer/global-ignorelist.txt

#![crate_type = "lib"]

// CHECK: @IGNORED_GLOBAL = {{.*}} no_sanitize_address
#[no_mangle]
pub static IGNORED_GLOBAL: i64 = 42;

// CHECK: @CHECKED_GLOBAL = {{.*}} no_sanitize_address
// (because of src:*global-ignorelist.rs)
#[no_mangle]
pub static CHECKED_GLOBAL: i64 = 42;
