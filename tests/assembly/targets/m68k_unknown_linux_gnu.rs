// assembly-output: emit-asm
// compile-flags: --target m68k-unknown-linux-gnu
// needs-llvm-components: m68k

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
