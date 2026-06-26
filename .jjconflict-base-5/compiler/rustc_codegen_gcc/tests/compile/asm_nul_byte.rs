// Compiler:
//   status: error
//   stderr:
//     error: asm contains a NUL byte
//     ...

// Test that inline asm containing a NUL byte emits an error.

use std::arch::asm;

fn main() {
    unsafe {
        asm!("\0");
    }
}
