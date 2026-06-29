//@ needs-sanitizer-address
//@ compile-flags: -Zsanitizer=address -Zsanitizer-ignorelist={{src-base}}/sanitizer/ignorelist.txt

#![crate_type = "lib"]

// CHECK: @IGNORED_GLOBAL = {{.*}} no_sanitize_address
#[no_mangle]
pub static IGNORED_GLOBAL: i32 = 42;

// CHECK: @CHECKED_GLOBAL =
// CHECK-NOT: no_sanitize_address
#[no_mangle]
pub static CHECKED_GLOBAL: i64 = 42;

pub struct MyStruct {
    x: i32,
}

// CHECK: @MY_STRUCT = {{.*}} no_sanitize_address
#[no_mangle]
pub static MY_STRUCT: MyStruct = MyStruct { x: 42 };
