//@ only-aarch64
//@ run-pass
//@ needs-asm-support

// Test that we properly work around this LLVM issue:
// https://github.com/llvm/llvm-project/issues/58384

use std::arch::asm;

fn main() {
    let a: i32;
    unsafe {
        asm!("", inout("x0") 435 => a);
    }
    assert_eq!(a, 435);
}
