//@ revisions: aarch64 x86_64
//@ assembly-output: emit-asm
//@[aarch64] only-aarch64
//@[x86_64] only-x86_64

#![feature(breakpoint)]
#![crate_type = "lib"]

// CHECK-LABEL: use_bp
// aarch64: brk #0xf000
// x86_64: int3
#[inline(never)]
pub fn use_bp() {
    core::arch::breakpoint();
}
