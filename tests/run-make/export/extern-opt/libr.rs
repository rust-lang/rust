#![feature(export)]
#![crate_type = "sdylib"]

#[export]
pub extern "C" fn foo(x: i32) -> i32 { x }
