// Compiler:
//   status: error
//   stderr:
//     ...
//     error: asm contains a NUL byte
//     ...

// Test that naked_asm containing a NUL byte emits an error.

#![crate_type = "lib"]

use std::arch::naked_asm;

#[unsafe(naked)]
pub extern "C" fn nul_byte_naked() {
    naked_asm!("\0")
}
