//@ add-core-stubs
//@ compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
//@ needs-llvm-components: arm
//@ check-pass

#![feature(cmse_nonsecure_entry, no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

#[repr(Rust)]
pub union ReprRustUnionU64 {
    _unused: u64,
}

#[no_mangle]
#[allow(improper_ctypes_definitions)]
pub extern "cmse-nonsecure-entry" fn union_rust() -> ReprRustUnionU64 {
    ReprRustUnionU64 { _unused: 1 }
    //~^ WARN passing a union across the security boundary may leak information
}

#[no_mangle]
pub extern "cmse-nonsecure-entry" fn maybe_uninit_32bit() -> MaybeUninit<u32> {
    MaybeUninit::uninit()
    //~^ WARN passing a union across the security boundary may leak information
}

#[no_mangle]
pub extern "cmse-nonsecure-entry" fn maybe_uninit_64bit() -> MaybeUninit<f64> {
    if true {
        return MaybeUninit::new(6.28);
        //~^ WARN passing a union across the security boundary may leak information
    }
    MaybeUninit::new(3.14)
    //~^ WARN passing a union across the security boundary may leak information
}

#[repr(transparent)]
pub struct Wrapper(ReprRustUnionU64);

#[no_mangle]
pub extern "cmse-nonsecure-entry" fn repr_transparent_union() -> Wrapper {
    //~^ WARN improper_ctypes_definitions
    match 0 {
        //~^ WARN passing a union across the security boundary may leak information
        0 => Wrapper(ReprRustUnionU64 { _unused: 1 }),
        _ => Wrapper(ReprRustUnionU64 { _unused: 2 }),
    }
}
