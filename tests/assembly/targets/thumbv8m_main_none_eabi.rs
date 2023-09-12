// assembly-output: emit-asm
// compile-flags: --target thumbv8m.main-none-eabi
// needs-llvm-components: arm

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_name = "thumbv8m_main_none_eabi"]
#![crate_type = "lib"]

#[lang = "sized"]
trait Sized {}

pub fn test() -> u8 {
    42
}

// CHECK: .section
