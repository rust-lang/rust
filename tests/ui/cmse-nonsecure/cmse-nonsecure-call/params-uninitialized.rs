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

#[repr(C)]
union ReprCUnionU64 {
    _unused: u64,
    _unused1: u32,
}

#[no_mangle]
fn basic(
    f1: extern "cmse-nonsecure-call" fn(ReprRustUnionU64),
    f2: extern "cmse-nonsecure-call" fn(ReprCUnionU64),
    f3: extern "cmse-nonsecure-call" fn(MaybeUninit<u32>),
    f4: extern "cmse-nonsecure-call" fn(MaybeUninit<u64>),
) {
    f1(ReprRustUnionU64 { _unused: 1 });
    //~^ WARN value crossing a secure boundary contains a union which may leak secure information

    f2(ReprCUnionU64 { _unused: 1 });
    //~^ WARN value crossing a secure boundary contains a union which may leak secure information

    f3(MaybeUninit::uninit());
    //~^ WARN value crossing a secure boundary contains a union which may leak secure information

    f4(MaybeUninit::uninit());
    //~^ WARN value crossing a secure boundary contains a union which may leak secure information
}

#[repr(C)]
struct ReprCAggregate {
    a: usize,
    b: ReprCUnionU64,
}

#[repr(Rust)]
union ReprRustUnionPartiallyUninit {
    _unused1: u32,
    _unused2: u16,
}

#[no_mangle]
fn composite(
    f5: extern "cmse-nonsecure-call" fn((usize, MaybeUninit<u64>)),
    f6: extern "cmse-nonsecure-call" fn(ReprCAggregate),
    f7: extern "cmse-nonsecure-call" fn(ReprRustUnionPartiallyUninit),
) {
    f5((0, MaybeUninit::uninit()));
    //~^ WARN value crossing a secure boundary contains a union which may leak secure information

    f6(ReprCAggregate { a: 0, b: ReprCUnionU64 { _unused: 1 } });
    //~^ WARN value crossing a secure boundary contains a union which may leak secure information

    f7(ReprRustUnionPartiallyUninit { _unused1: 0 });
    //~^ WARN value crossing a secure boundary contains a union which may leak secure information
}

#[no_mangle]
fn nested(
    f12: extern "cmse-nonsecure-call" fn(Option<MaybeUninit<u32>>),
    f13: extern "cmse-nonsecure-call" fn(Result<MaybeUninit<u32>, ReprCUnionU64>),
) {
    f12(Some(MaybeUninit::uninit()));
    //~^ WARN value crossing a secure boundary contains a union which may leak secure information

    f13(Ok(MaybeUninit::new(42)));
    //~^ WARN value crossing a secure boundary contains a union which may leak secure information

    f13(Err(ReprCUnionU64 { _unused1: 0 }));
    //~^ WARN value crossing a secure boundary contains a union which may leak secure information
}

struct ArrayVec<T, const N: usize>([MaybeUninit<T>; N]);

#[no_mangle]
fn array_vec(
    f0: extern "cmse-nonsecure-call" fn(ArrayVec<u8, 8>),
    f1: extern "cmse-nonsecure-call" fn(ArrayVec<u8, 0>),
) {
    f0(ArrayVec([MaybeUninit::uninit(); 8]));
    //~^ WARN value crossing a secure boundary contains a union which may leak secure information

    f0(ArrayVec([MaybeUninit::new(0xAA); 8]));
    //~^ WARN value crossing a secure boundary contains a union which may leak secure information

    f1(ArrayVec([MaybeUninit::uninit(); 0]));
    //~^ WARN value crossing a secure boundary contains a union which may leak secure information
}
