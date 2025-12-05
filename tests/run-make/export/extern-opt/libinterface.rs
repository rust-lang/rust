#![feature(export_stable)]

#[export_stable]
pub extern "C" fn foo(x: i32) -> i32;
