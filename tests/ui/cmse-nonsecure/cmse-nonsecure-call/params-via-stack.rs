//@ build-fail
//@ compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
//@ needs-llvm-components: arm
#![feature(abi_c_cmse_nonsecure_call, no_core, lang_items, intrinsics)]
#![no_core]
#[lang = "sized"]
pub trait Sized {}
#[lang = "copy"]
pub trait Copy {}
impl Copy for u32 {}

#[repr(C, align(16))]
#[allow(unused)]
pub struct AlignRelevant(u32);

#[no_mangle]
pub fn test(
    f1: extern "C-cmse-nonsecure-call" fn(u32, u32, u32, u32, u32) -> u32,
    f2: extern "C-cmse-nonsecure-call" fn(u32, u32, u32, u16, u16) -> u32,
    f3: extern "C-cmse-nonsecure-call" fn(u32, u64, u32) -> u32,
    f4: extern "C-cmse-nonsecure-call" fn(AlignRelevant, u32) -> u32,
    f5: extern "C-cmse-nonsecure-call" fn([u32; 5]) -> u32,
) {
    f1(1, 2, 3, 4, 5); //~ ERROR [E0798]
    f2(1, 2, 3, 4, 5); //~ ERROR [E0798]
    f3(1, 2, 3); //~ ERROR [E0798]
    f4(AlignRelevant(1), 2); //~ ERROR [E0798]
    f5([0xAA; 5]); //~ ERROR [E0798]
}
