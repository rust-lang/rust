// assembly-output: emit-asm
// compile-flags: --target powerpc64-wrs-vxworks
// needs-llvm-components: powerpc

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
