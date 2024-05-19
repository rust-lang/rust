//@ build-pass
//@ compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
//@ needs-llvm-components: arm
#![feature(cmse_nonsecure_entry, no_core, lang_items)]
#![no_core]
#[lang="sized"]
trait Sized { }
#[lang="copy"]
trait Copy { }
impl Copy for u32 {}

#[no_mangle]
#[cmse_nonsecure_entry]
pub extern "C" fn entry_function(_: u32, _: u32, _: u32, d: u32) -> u32 {
    d
}
