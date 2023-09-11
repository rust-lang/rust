// assembly-output: emit-asm
// compile-flags: --target aarch64_be-unknown-linux-gnu_ilp32
// needs-llvm-components: aarch64

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
