// assembly-output: emit-asm
// compile-flags: --target thumbv7a-uwp-windows-msvc
// needs-llvm-components: arm

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

#[lang = "sized"]
trait Sized {}

pub fn test() -> u8 {
    42
}

// CHECK: .text
