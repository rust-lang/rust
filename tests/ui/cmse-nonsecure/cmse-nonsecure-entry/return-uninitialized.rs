//@ add-minicore
//@ compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
//@ needs-llvm-components: arm
//@ check-pass
//@ ignore-backends: gcc

#![feature(cmse_nonsecure_entry, no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

#[repr(Rust)]
union ReprRustUnionU32 {
    _unused: u32,
}

#[no_mangle]
#[allow(improper_ctypes_definitions)]
extern "cmse-nonsecure-entry" fn union_rust() -> ReprRustUnionU32 {
    ReprRustUnionU32 { _unused: 1 }
    //~^ WARN value crossing a secure boundary contains a union which may leak secure information
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
    //~^ WARN value crossing a secure boundary contains a union which may leak secure information
}

#[no_mangle]
extern "cmse-nonsecure-entry" fn maybe_uninit_32bit() -> MaybeUninit<u32> {
    MaybeUninit::uninit()
    //~^ WARN value crossing a secure boundary contains a union which may leak secure information
}

#[no_mangle]
extern "cmse-nonsecure-entry" fn maybe_uninit_64bit() -> MaybeUninit<f64> {
    if true {
        return MaybeUninit::new(6.28);
        //~^ WARN value crossing a secure boundary contains a union which may leak secure information
    }
    MaybeUninit::new(3.14)
    //~^ WARN value crossing a secure boundary contains a union which may leak secure information
}

#[repr(transparent)]
struct Wrapper(MaybeUninit<u64>);

#[no_mangle]
extern "cmse-nonsecure-entry" fn repr_transparent_union() -> Wrapper {
    match 0 {
        //~^ WARN value crossing a secure boundary contains a union which may leak secure information
        0 => Wrapper(MaybeUninit::new(0)),
        _ => Wrapper(MaybeUninit::new(1)),
    }
}

#[no_mangle]
#[allow(improper_ctypes_definitions)]
extern "cmse-nonsecure-entry" fn option_maybe_uninit() -> Option<MaybeUninit<u8>> {
    None
    //~^ WARN value crossing a secure boundary contains a union which may leak secure information
}

#[repr(C)]
struct ArrayVec<T, const N: usize>([MaybeUninit<T>; N]);

#[no_mangle]
extern "cmse-nonsecure-entry" fn array_vec_single(x: u8) -> ArrayVec<u8, 2> {
    ArrayVec([MaybeUninit::new(x), MaybeUninit::uninit()])
    //~^ WARN value crossing a secure boundary contains a union which may leak secure information
}

#[no_mangle]
extern "cmse-nonsecure-entry" fn array_vec_zero_sized() -> ArrayVec<u8, 0> {
    ArrayVec([])
    //~^ WARN value crossing a secure boundary contains a union which may leak secure information
}
