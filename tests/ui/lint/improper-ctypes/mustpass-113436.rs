//@ check-pass
#![deny(improper_ctypes_definitions, improper_ctypes)]

#[repr(C)]
pub struct Wrap<T>(T);

#[repr(transparent)]
pub struct TransparentWrap<T>(T);

pub extern "C" fn f() -> Wrap<u8> {
    todo!()
}

const _: extern "C" fn() -> Wrap<u8> = f;

pub extern "C" fn ff() -> Wrap<Wrap<u8>> {
    todo!()
}

const _: extern "C" fn() -> Wrap<Wrap<u8>> = ff;

pub extern "C" fn g() -> TransparentWrap<u8> {
    todo!()
}

const _: extern "C" fn() -> TransparentWrap<u8> = g;

pub extern "C" fn gg() -> TransparentWrap<TransparentWrap<u8>> {
    todo!()
}

const _: extern "C" fn() -> TransparentWrap<TransparentWrap<u8>> = gg;

fn main() {}
