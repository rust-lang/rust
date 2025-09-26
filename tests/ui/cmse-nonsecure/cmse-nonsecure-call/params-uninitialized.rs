//@ add-core-stubs
//@ compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
//@ needs-llvm-components: arm
//@ check-pass
#![feature(abi_cmse_nonsecure_call, no_core, lang_items)]
#![no_core]
#![allow(improper_ctypes_definitions)]

extern crate minicore;
use minicore::*;

#[repr(Rust)]
pub union ReprRustUnionU64 {
    _unused: u64,
}

#[repr(C)]
pub union ReprCUnionU64 {
    _unused: u64,
    _unused1: u32,
}

#[repr(C)]
pub struct ReprCAggregate {
    a: usize,
    b: ReprCUnionU64,
}

#[no_mangle]
pub fn test_union(
    f1: extern "cmse-nonsecure-call" fn(ReprRustUnionU64),
    f2: extern "cmse-nonsecure-call" fn(ReprCUnionU64),
    f3: extern "cmse-nonsecure-call" fn(MaybeUninit<u32>),
    f4: extern "cmse-nonsecure-call" fn(MaybeUninit<u64>),
    f5: extern "cmse-nonsecure-call" fn((usize, MaybeUninit<u64>)),
    f6: extern "cmse-nonsecure-call" fn(ReprCAggregate),
) {
    f1(ReprRustUnionU64 { _unused: 1 });
    //~^ WARN passing a union across the security boundary may leak information

    f2(ReprCUnionU64 { _unused: 1 });
    //~^ WARN passing a union across the security boundary may leak information

    f3(MaybeUninit::uninit());
    //~^ WARN passing a union across the security boundary may leak information

    f4(MaybeUninit::uninit());
    //~^ WARN passing a union across the security boundary may leak information

    f5((0, MaybeUninit::uninit()));
    //~^ WARN passing a union across the security boundary may leak information

    f6(ReprCAggregate { a: 0, b: ReprCUnionU64 { _unused: 1 } });
    //~^ WARN passing a union across the security boundary may leak information
}
