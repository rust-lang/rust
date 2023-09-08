// assembly-output: emit-asm
// compile-flags: --target mipsisa32r6-unknown-linux-gnu
// needs-llvm-components: mips

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

#[lang = "sized"]
trait Sized {}

pub fn test() -> u8 {
    42
}

// CHECK: .section
