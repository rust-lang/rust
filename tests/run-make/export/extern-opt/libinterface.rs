#![feature(export)]

#[export]
pub extern "C" fn foo(x: i32) -> i32;
