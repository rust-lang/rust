//@ needs-sanitizer-address
//@ compile-flags: -Zsanitizer=address -Zsanitizer-ignorelist={{src-base}}/sanitizer/type-string-unsafe-ignorelist.txt

#![crate_type = "lib"]

pub static MY_FN: unsafe extern "C" fn() = my_fn_impl;

// CHECK: MY_FN = {{.*}} no_sanitize_address

unsafe extern "C" fn my_fn_impl() {}
