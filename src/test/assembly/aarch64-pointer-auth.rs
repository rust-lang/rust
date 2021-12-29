// Test that PAC instructions are emitted when branch-protection is specified.

// min-llvm-version: 10.0.1
// assembly-output: emit-asm
// compile-flags: --target aarch64-unknown-linux-gnu
// compile-flags: -Z branch-protection=pac-ret,leaf
// needs-llvm-components: aarch64

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

#[lang = "sized"]
trait Sized {}

// CHECK: hint #25
// CHECK: hint #29
#[no_mangle]
pub fn test() -> u8 {
    42
}
