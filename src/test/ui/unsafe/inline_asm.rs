// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

#![feature(llvm_asm)]
#![feature(asm)]

fn main() {
    asm!("nop"); //~ ERROR use of inline assembly is unsafe and requires unsafe function or block
    llvm_asm!("nop"); //~ ERROR use of inline assembly is unsafe and requires unsafe function or block
}
