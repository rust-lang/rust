#![crate_type = "cdylib"]

extern crate bar;

#[no_mangle]
pub extern "C" fn foo() {
    bar::bar();
}

#[no_mangle]
pub extern "C" fn bar(a: u32, b: u32) -> u32 {
    a + b
}
