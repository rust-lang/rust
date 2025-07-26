//@ add-core-stubs
//@ build-pass
//@ compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
//@ needs-llvm-components: arm
#![feature(cmse_nonsecure_entry, no_core, lang_items)]
#![no_core]
#![crate_type = "lib"]

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
pub extern "cmse-nonsecure-entry" fn inputs1() {}
#[no_mangle]
pub extern "cmse-nonsecure-entry" fn inputs2(_: u32, _: u32, _: u32, _: u32) {}
#[no_mangle]
pub extern "cmse-nonsecure-entry" fn inputs3(_: u64, _: u64) {}
#[no_mangle]
#[allow(improper_ctypes_definitions)]
pub extern "cmse-nonsecure-entry" fn inputs4(_: u128) {}
#[no_mangle]
pub extern "cmse-nonsecure-entry" fn inputs5(_: f64, _: f32, _: f32) {}
#[no_mangle]
pub extern "cmse-nonsecure-entry" fn inputs6(_: ReprTransparentStruct<u64>, _: U32Compound) {}
#[no_mangle]
#[allow(improper_ctypes_definitions)]
pub extern "cmse-nonsecure-entry" fn inputs7(_: [u32; 4]) {}

#[no_mangle]
pub extern "cmse-nonsecure-entry" fn outputs1() -> u32 {
    0
}
#[no_mangle]
pub extern "cmse-nonsecure-entry" fn outputs2() -> u64 {
    0
}
#[no_mangle]
pub extern "cmse-nonsecure-entry" fn outputs3() -> i64 {
    0
}
#[no_mangle]
pub extern "cmse-nonsecure-entry" fn outputs4() -> f64 {
    0.0
}
#[no_mangle]
#[allow(improper_ctypes_definitions)]
pub extern "cmse-nonsecure-entry" fn outputs5() -> [u8; 4] {
    [0xAA; 4]
}
#[no_mangle]
pub extern "cmse-nonsecure-entry" fn outputs6() -> ReprTransparentStruct<u64> {
    ReprTransparentStruct { _marker1: (), _marker2: (), field: 0xAA, _marker3: () }
}
#[no_mangle]
pub extern "cmse-nonsecure-entry" fn outputs7(
) -> ReprTransparentStruct<ReprTransparentStruct<u64>> {
    ReprTransparentStruct {
        _marker1: (),
        _marker2: (),
        field: ReprTransparentStruct { _marker1: (), _marker2: (), field: 0xAA, _marker3: () },
        _marker3: (),
    }
}
#[no_mangle]
pub extern "cmse-nonsecure-entry" fn outputs8() -> ReprTransparentEnumU64 {
    ReprTransparentEnumU64::A(0)
}
#[no_mangle]
pub extern "cmse-nonsecure-entry" fn outputs9() -> U32Compound {
    U32Compound(1, 2)
}
