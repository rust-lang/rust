//@ build-pass
//@ compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
//@ needs-llvm-components: arm
#![feature(abi_c_cmse_nonsecure_call, no_core, lang_items, intrinsics)]
#![no_core]
#[lang = "sized"]
pub trait Sized {}
#[lang = "copy"]
pub trait Copy {}
impl Copy for u32 {}

#[repr(transparent)]
pub struct ReprTransparentStructU64 {
    _marker1: (),
    _marker2: (),
    field: u64,
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
    f1: extern "C-cmse-nonsecure-call" fn(),
    f2: extern "C-cmse-nonsecure-call" fn(u32, u32, u32, u32),
    f3: extern "C-cmse-nonsecure-call" fn(u64, u64),
    f4: extern "C-cmse-nonsecure-call" fn(u128),
    f5: extern "C-cmse-nonsecure-call" fn(f64, f32, f32),
    f6: extern "C-cmse-nonsecure-call" fn(ReprTransparentStructU64, U32Compound),
    f7: extern "C-cmse-nonsecure-call" fn([u32; 4]),
) {
    f1();
    f2(1, 2, 3, 4);
    f3(1, 2);
    f4(1);
    f5(1.0, 2.0, 3.0);
    f6(
        ReprTransparentStructU64 { _marker1: (), _marker2: (), field: 1, _marker3: () },
        U32Compound(2, 3),
    );
    f7([0xDEADBEEF; 4]);
}

#[no_mangle]
pub fn returns(
    f1: extern "C-cmse-nonsecure-call" fn() -> u32,
    f2: extern "C-cmse-nonsecure-call" fn() -> u64,
    f3: extern "C-cmse-nonsecure-call" fn() -> i64,
    f4: extern "C-cmse-nonsecure-call" fn() -> f64,
    f5: extern "C-cmse-nonsecure-call" fn() -> [u8; 4],
    f6: extern "C-cmse-nonsecure-call" fn() -> ReprTransparentStructU64,
    f7: extern "C-cmse-nonsecure-call" fn() -> ReprTransparentEnumU64,
    f8: extern "C-cmse-nonsecure-call" fn() -> U32Compound,
) {
    f1();
    f2();
    f3();
    f4();
    f5();
    f6();
    f7();
    f8();
}
