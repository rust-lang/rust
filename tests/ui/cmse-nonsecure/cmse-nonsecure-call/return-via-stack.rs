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

#[repr(C)]
pub struct ReprCU64(u64);

#[repr(C)]
pub struct ReprCBytes(u8, u8, u8, u8, u8);

#[repr(C)]
pub struct U64Compound(u32, u32);

#[repr(C, align(16))]
pub struct ReprCAlign16(u16);

#[no_mangle]
pub fn test(
    f1: extern "C-cmse-nonsecure-call" fn() -> ReprCU64,
    f2: extern "C-cmse-nonsecure-call" fn() -> ReprCBytes,
    f3: extern "C-cmse-nonsecure-call" fn() -> U64Compound,
    f4: extern "C-cmse-nonsecure-call" fn() -> ReprCAlign16,
    f5: extern "C-cmse-nonsecure-call" fn() -> [u8; 5],
    f6: extern "C-cmse-nonsecure-call" fn() -> u128, //~ WARNING [improper_ctypes_definitions]
    f7: extern "C-cmse-nonsecure-call" fn() -> i128, //~ WARNING [improper_ctypes_definitions]
) {
    f1(); //~ ERROR [E0798]
    f2(); //~ ERROR [E0798]
    f3(); //~ ERROR [E0798]
    f4(); //~ ERROR [E0798]
    f5(); //~ ERROR [E0798]
    f6(); //~ ERROR [E0798]
    f7(); //~ ERROR [E0798]
}
