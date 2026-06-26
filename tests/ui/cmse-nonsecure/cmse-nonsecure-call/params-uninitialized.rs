//@ add-minicore
//@ compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
//@ needs-llvm-components: arm
//@ check-pass
//@ ignore-backends: gcc
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

#[repr(C)]
enum VariantsSameSize {
    A(u16),
    B(u16),
}

#[repr(C)]
enum VariantsDifferentSize {
    A(u8),
    B(u16),
}

enum Void {}

#[repr(C)]
enum UninhabitedVariant {
    A(Void),
    B(u16),
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
    f9: extern "cmse-nonsecure-call" fn(VariantsSameSize),
    f10: extern "cmse-nonsecure-call" fn(VariantsDifferentSize),
    f11: extern "cmse-nonsecure-call" fn(UninhabitedVariant),
) {
    // With `repr(Rust)` this union is always initialized.
    f1(ReprRustUnionU64 { _unused: 1 });

    f2(ReprCUnionU64 { _unused: 1 });
    //~^ WARN this value crossing a secure boundary may contain (partially) uninitialized data which can leak information

    f3(MaybeUninit::uninit());
    //~^ WARN this value crossing a secure boundary may contain (partially) uninitialized data which can leak information

    f4(MaybeUninit::uninit());
    //~^ WARN this value crossing a secure boundary may contain (partially) uninitialized data which can leak information

    f5((0, MaybeUninit::uninit()));
    //~^ WARN this value crossing a secure boundary may contain (partially) uninitialized data which can leak information

    f6(ReprCAggregate { a: 0, b: ReprCUnionU64 { _unused: 1 } });
    //~^ WARN this value crossing a secure boundary may contain (partially) uninitialized data which can leak information

    f7(ReprRustUnionPartiallyUninit { _unused1: 0 });
    //~^ WARN this value crossing a secure boundary may contain (partially) uninitialized data which can leak information

    // This struct only has no value-dependent padding, the guaranteed padding is zeroed.
    f8(PaddedStruct { a: 0, b: 0 });

    // This enum only has no value-dependent padding, the guaranteed padding is zeroed.
    f9(VariantsSameSize::A(0));

    f10(VariantsDifferentSize::A(0));
    //~^ WARN this value crossing a secure boundary may contain (partially) uninitialized data which can leak information

    f11(UninhabitedVariant::B(0));
    //~^ WARN this value crossing a secure boundary may contain (partially) uninitialized data which can leak information
}
