#![feature(export_stable)]
#![crate_type = "sdylib"]

#[export_stable]
pub extern "C" fn foo(x: i32) -> i32 { x }
