//@ needs-asm-support
//@ run-pass

#![feature(global_asm_statement_position)]

use std::arch::global_asm;

fn main() {
    global_asm!("");
}
