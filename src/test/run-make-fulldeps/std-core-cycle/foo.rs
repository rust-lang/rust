#![crate_type = "cdylib"]

extern crate bar;

#[global_allocator]
static A: bar::A = bar::A;

#[no_mangle]
pub extern fn a(a: u32, b: u32) -> u32 {
    a / b
}
