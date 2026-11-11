//@ assembly-output: emit-asm
//@ only-x86_64-unknown-linux-gnu

#![feature(used_with_arg)]
#![crate_type = "lib"]

// CHECK: .section	.rodata.X,"a"
#[used(compiler)]
#[no_mangle]
pub static X: u32 = 12;
// CHECK: .section	.rodata.Y,"aR"
#[used(linker)]
#[no_mangle]
pub static Y: u32 = 12;
