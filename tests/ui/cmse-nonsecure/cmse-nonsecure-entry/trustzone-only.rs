//@ revisions: x86 aarch64 thumb7
//
//@[x86] compile-flags: --target x86_64-unknown-linux-gnu
//@[x86] needs-llvm-components: x86
//@[aarch64] compile-flags: --target aarch64-unknown-linux-gnu
//@[aarch64] needs-llvm-components: aarch64
//@[thumb7] compile-flags: --target thumbv7em-none-eabi
//@[thumb7] needs-llvm-components: arm
#![feature(no_core, lang_items, rustc_attrs, cmse_nonsecure_entry)]
#![no_core]

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}

impl Copy for u32 {}

#[no_mangle]
pub extern "C-cmse-nonsecure-entry" fn entry_function(input: u32) -> u32 {
    //~^ ERROR [E0570]
    input
}

fn main() {}
