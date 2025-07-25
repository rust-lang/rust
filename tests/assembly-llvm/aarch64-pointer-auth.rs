// Test that PAC instructions are emitted when branch-protection is specified.

//@ add-core-stubs
//@ revisions: PACRET PAUTHLR_NOP PAUTHLR
//@ assembly-output: emit-asm
//@ needs-llvm-components: aarch64
//@ compile-flags: --target aarch64-unknown-linux-gnu
//@ [PACRET] compile-flags: -Z branch-protection=pac-ret,leaf
//@ [PAUTHLR_NOP] compile-flags: -Z branch-protection=pac-ret,pc,leaf
//@ [PAUTHLR] compile-flags: -C target-feature=+pauth-lr -Z branch-protection=pac-ret,pc,leaf

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

// PACRET: hint #25
// PACRET: hint #29
// PAUTHLR_NOP: hint #25
// PAUTHLR_NOP: hint #39
// PAUTHLR_NOP: hint #29
// PAUTHLR: paciasppc
// PAUTHLR: autiasppc
#[no_mangle]
pub fn test() -> u8 {
    42
}
