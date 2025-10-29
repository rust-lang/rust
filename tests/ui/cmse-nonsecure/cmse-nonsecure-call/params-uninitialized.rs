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
union ReprRustUnionU64 {
    _unused: u64,
}

#[repr(Rust)]
union ReprRustUnionPartiallyUninit {
    _unused1: u32,
    _unused2: u16,
}

#[repr(C)]
union ReprCUnionU64 {
    _unused: u64,
    _unused1: u32,
}

#[repr(C)]
struct ReprCAggregate {
    a: usize,
    b: ReprCUnionU64,
}

// This is an aggregate that cannot be unwrapped, and has 1 (uninitialized) padding byte.
#[repr(C, align(4))]
struct PaddedStruct {
    a: u8,
    b: u16,
}

#[no_mangle]
fn test_uninitialized(
    f1: extern "cmse-nonsecure-call" fn(ReprRustUnionU64),
    f2: extern "cmse-nonsecure-call" fn(ReprCUnionU64),
    f3: extern "cmse-nonsecure-call" fn(MaybeUninit<u32>),
    f4: extern "cmse-nonsecure-call" fn(MaybeUninit<u64>),
    f5: extern "cmse-nonsecure-call" fn((usize, MaybeUninit<u64>)),
    f6: extern "cmse-nonsecure-call" fn(ReprCAggregate),
    f7: extern "cmse-nonsecure-call" fn(ReprRustUnionPartiallyUninit),
    f8: extern "cmse-nonsecure-call" fn(PaddedStruct),
) {
    // With `repr(Rust)` this union is always initialized.
    f1(ReprRustUnionU64 { _unused: 1 });

    f2(ReprCUnionU64 { _unused: 1 });
    //~^ WARN passing a (partially) uninitialized value across the security boundary may leak information

    f3(MaybeUninit::uninit());
    //~^ WARN passing a (partially) uninitialized value across the security boundary may leak information

    f4(MaybeUninit::uninit());
    //~^ WARN passing a (partially) uninitialized value across the security boundary may leak information

    f5((0, MaybeUninit::uninit()));
    //~^ WARN passing a (partially) uninitialized value across the security boundary may leak information

    f6(ReprCAggregate { a: 0, b: ReprCUnionU64 { _unused: 1 } });
    //~^ WARN passing a (partially) uninitialized value across the security boundary may leak information

    f7(ReprRustUnionPartiallyUninit { _unused1: 0 });
    //~^ WARN passing a (partially) uninitialized value across the security boundary may leak information

    f8(PaddedStruct { a: 0, b: 0 });
    //~^ WARN passing a (partially) uninitialized value across the security boundary may leak information
}
