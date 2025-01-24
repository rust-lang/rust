//@ build-pass
//@ compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
//@ needs-llvm-components: arm
#![feature(abi_c_cmse_nonsecure_call, cmse_nonsecure_entry, no_core, lang_items, intrinsics)]
#![no_core]
#[lang = "sized"]
pub trait Sized {}
#[lang = "copy"]
pub trait Copy {}
impl Copy for u32 {}

#[no_mangle]
pub extern "C-cmse-nonsecure-entry" fn test(
    f: extern "C-cmse-nonsecure-call" fn(u32, u32, u32, u32) -> u32,
    a: u32,
    b: u32,
    c: u32,
) -> u32 {
    f(a, b, c, 42)
}
