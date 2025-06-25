//@ add-core-stubs
//@ build-pass
//@ compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
//@ needs-llvm-components: arm
#![feature(abi_cmse_nonsecure_call, no_core, lang_items, intrinsics)]
#![no_core]

extern crate minicore;
use minicore::*;

#[repr(transparent)]
pub struct ReprTransparentStruct<T> {
    _marker1: (),
    _marker2: (),
    field: T,
    _marker3: (),
}

#[repr(transparent)]
pub enum ReprTransparentEnumU64 {
    A(u64),
}

#[repr(C)]
pub struct U32Compound(u16, u16);

#[no_mangle]
#[allow(improper_ctypes_definitions)]
pub fn params(
    f1: extern "cmse-nonsecure-call" fn(),
    f2: extern "cmse-nonsecure-call" fn(u32, u32, u32, u32),
    f3: extern "cmse-nonsecure-call" fn(u64, u64),
    f4: extern "cmse-nonsecure-call" fn(u128),
    f5: extern "cmse-nonsecure-call" fn(f64, f32, f32),
    f6: extern "cmse-nonsecure-call" fn(ReprTransparentStruct<u64>, U32Compound),
    f7: extern "cmse-nonsecure-call" fn([u32; 4]),
) {
}

#[no_mangle]
pub fn returns(
    f1: extern "cmse-nonsecure-call" fn() -> u32,
    f2: extern "cmse-nonsecure-call" fn() -> u64,
    f3: extern "cmse-nonsecure-call" fn() -> i64,
    f4: extern "cmse-nonsecure-call" fn() -> f64,
    f5: extern "cmse-nonsecure-call" fn() -> [u8; 4],
    f6: extern "cmse-nonsecure-call" fn() -> ReprTransparentStruct<u64>,
    f7: extern "cmse-nonsecure-call" fn() -> ReprTransparentStruct<ReprTransparentStruct<u64>>,
    f8: extern "cmse-nonsecure-call" fn() -> ReprTransparentEnumU64,
    f9: extern "cmse-nonsecure-call" fn() -> U32Compound,
) {
}
