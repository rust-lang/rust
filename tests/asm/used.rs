//@ assembly-output: emit-asm
//@ only-x86_64-unknown-linux-gnu

#![feature(used_with_arg)]
#![crate_type = "lib"]

// CHECK: .section	.rodata._RNvCslVCd7eQSKhE_4used1X,"a"
#[used(compiler)]
pub static X: u32 = 12;
// CHECK: .section	.rodata._RNvCslVCd7eQSKhE_4used1Y,"aR"
#[used(linker)]
pub static Y: u32 = 12;
