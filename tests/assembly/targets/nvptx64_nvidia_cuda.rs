// assembly-output: emit-asm
// compile-flags: --target nvptx64-nvidia-cuda
// needs-llvm-components: nvptx

#![feature(no_core, lang_items)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

#[lang = "sized"]
trait Sized {}

pub fn test() -> u8 {
    42
}

// CHECK: .version
