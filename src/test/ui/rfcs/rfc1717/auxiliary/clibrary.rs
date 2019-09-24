// no-prefer-dynamic
#![crate_type = "staticlib"]

#[no_mangle]
pub extern "C" fn foo(x:i32) -> i32 { x }
