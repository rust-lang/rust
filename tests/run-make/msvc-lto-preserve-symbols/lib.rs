#![feature(no_core)]
#![no_core]

extern crate minicore;

// Ordinary exported names must not be replaced with their COFF linker
// spellings in the LTO preservation list.
#[no_mangle]
pub extern "C" fn __rust_lto_preserve_plain(x: u32) -> u32 {
    x
}

// Explicit calling-convention decoration must survive both ordinary LTO
// internalization and ThinLTO GUID-based preservation.
#[no_mangle]
pub extern "stdcall" fn rust_lto_preserve_stdcall(a: u32, _: u32) -> u32 {
    a
}

#[no_mangle]
pub static RUST_LTO_PRESERVE_DATA: u32 = 42;
