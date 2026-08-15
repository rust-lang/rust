//@ needs-asm-support
//@ ignore-nvptx64
//@ ignore-spirv
//@ build-pass

#![feature(asm_const_ptr)]

use std::arch::{asm, global_asm};
use std::ptr::addr_of;

static FOO: u8 = 42;

global_asm!("/* {} */", const addr_of!(FOO));

#[no_mangle]
fn inline() {
    unsafe { asm!("/* {} */", const addr_of!(FOO)) };
}

fn main() {}
