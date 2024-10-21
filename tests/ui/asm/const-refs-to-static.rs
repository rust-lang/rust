//@ needs-asm-support
//@ ignore-nvptx64
//@ ignore-spirv

use std::arch::{asm, global_asm};
use std::ptr::addr_of;

static FOO: u8 = 42;

global_asm!("{}", const addr_of!(FOO));
//~^ ERROR invalid type for `const` operand

#[no_mangle]
fn inline() {
    unsafe { asm!("{}", const addr_of!(FOO)) };
    //~^ ERROR invalid type for `const` operand
}

fn main() {}
