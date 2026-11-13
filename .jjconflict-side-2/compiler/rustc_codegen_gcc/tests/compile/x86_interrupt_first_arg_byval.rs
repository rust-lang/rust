// Compiler:

// Test that `x86-interrupt` functions whose first argument is passed by value
// emit pointer-shaped GCC parameters and compile with interrupt-safe target features.

#![feature(abi_x86_interrupt)]
#![crate_type = "lib"]

#[repr(C)]
pub struct Frame {
    ip: u64,
}

pub extern "x86-interrupt" fn scalar(_a: i64) {}

pub extern "x86-interrupt" fn aggregate(_frame: Frame) {}
