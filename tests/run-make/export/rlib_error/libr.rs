#![feature(export)]
#![crate_type = "rlib"]

#[export]
pub extern "C" fn foo() -> i32 {
    42
}
