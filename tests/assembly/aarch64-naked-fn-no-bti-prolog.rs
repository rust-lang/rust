//@ compile-flags: -C no-prepopulate-passes -Zbranch-protection=bti
//@ assembly-output: emit-asm
//@ needs-asm-support
//@ only-aarch64

#![crate_type = "lib"]
#![feature(naked_functions)]
use std::arch::asm;

// The problem at hand: Rust has adopted a fairly strict meaning for "naked functions",
// meaning "no prologue whatsoever, no, really, not one instruction."
// Unfortunately, aarch64's "branch target identification" works via hints at landing sites.
// LLVM implements this via making sure of that, even for functions with the naked attribute.
// So, we must emit an appropriate instruction instead!
#[no_mangle]
#[naked]
pub unsafe extern "C" fn _hlt() -> ! {
    // CHECK-NOT: hint #34
    // CHECK: hlt #0x1
    asm!("hlt #1", options(noreturn))
}
