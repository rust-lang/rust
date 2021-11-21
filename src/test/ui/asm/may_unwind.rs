// min-llvm-version: 13.0.0
// run-pass
// needs-asm-support

#![feature(asm, asm_unwind)]

fn main() {
    unsafe { asm!("", options(may_unwind)) };
}
