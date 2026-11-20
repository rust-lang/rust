// Compiler:
//   status: error
//   stderr:
//     error: asm contains a NUL byte
//     ...

// Test that global_asm containing a NUL byte emits an error.

#![crate_type = "lib"]

use std::arch::global_asm;

global_asm!("\0");
