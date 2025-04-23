//@ compile-flags: -C no-prepopulate-passes -Zcf-protection=full
//@ assembly-output: emit-asm
//@ needs-asm-support
//@ only-x86_64

#![crate_type = "lib"]

use std::arch::naked_asm;

// The problem at hand: Rust has adopted a fairly strict meaning for "naked functions",
// meaning "no prologue whatsoever, no, really, not one instruction."
// Unfortunately, x86's control-flow enforcement, specifically indirect branch protection,
// works by using an instruction for each possible landing site,
// and LLVM implements this via making sure of that.
#[no_mangle]
#[unsafe(naked)]
pub extern "sysv64" fn will_halt() -> ! {
    // CHECK-NOT: endbr{{32|64}}
    // CHECK: hlt
    naked_asm!("hlt")
}

// what about aarch64?
// "branch-protection"=false
