// assembly-output: emit-asm
// compile-flags: --target s390x-unknown-linux-gnu
// needs-llvm-components: s390x

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
