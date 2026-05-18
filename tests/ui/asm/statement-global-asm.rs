//@ needs-asm-support
//@ run-pass

use std::arch::global_asm;

fn main() {
    global_asm!("");
}
