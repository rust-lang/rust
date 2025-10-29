//@ add-core-stubs
//@ compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
//@ needs-llvm-components: arm
//@ check-pass

#![feature(cmse_nonsecure_entry, no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

#[repr(Rust)]
union ReprRustUnionU64 {
    _unused: u64,
}

#[no_mangle]
#[allow(improper_ctypes_definitions)]
extern "cmse-nonsecure-entry" fn union_rust() -> ReprRustUnionU64 {
    // With `repr(Rust)` value is always fully initialized.
    ReprRustUnionU64 { _unused: 1 }
}

#[repr(Rust)]
union ReprRustUnionPartiallyUninit {
    _unused1: u32,
    _unused2: u16,
}

#[no_mangle]
#[allow(improper_ctypes_definitions)]
extern "cmse-nonsecure-entry" fn union_rust_partially_uninit() -> ReprRustUnionPartiallyUninit {
    ReprRustUnionPartiallyUninit { _unused1: 1 }
    //~^ WARN passing a (partially) uninitialized value across the security boundary may leak information
}

#[no_mangle]
extern "cmse-nonsecure-entry" fn maybe_uninit_32bit() -> MaybeUninit<u32> {
    MaybeUninit::uninit()
    //~^ WARN passing a (partially) uninitialized value across the security boundary may leak information
}

#[no_mangle]
extern "cmse-nonsecure-entry" fn maybe_uninit_64bit() -> MaybeUninit<f64> {
    if true {
        return MaybeUninit::new(6.28);
        //~^ WARN passing a (partially) uninitialized value across the security boundary may leak information
    }
    MaybeUninit::new(3.14)
    //~^ WARN passing a (partially) uninitialized value across the security boundary may leak information
}

#[repr(transparent)]
struct Wrapper(MaybeUninit<u64>);

#[no_mangle]
extern "cmse-nonsecure-entry" fn repr_transparent_union() -> Wrapper {
    match 0 {
        //~^ WARN passing a (partially) uninitialized value across the security boundary may leak information
        0 => Wrapper(MaybeUninit::new(0)),
        _ => Wrapper(MaybeUninit::new(1)),
    }
}

// This is an aggregate that cannot be unwrapped, and has 1 (uninitialized) padding byte.
#[repr(C, align(4))]
struct PaddedStruct {
    a: u8,
    b: u16,
}

#[no_mangle]
extern "cmse-nonsecure-entry" fn padded_struct() -> PaddedStruct {
    PaddedStruct { a: 0, b: 1 }
    //~^ WARN passing a (partially) uninitialized value across the security boundary may leak information
}
