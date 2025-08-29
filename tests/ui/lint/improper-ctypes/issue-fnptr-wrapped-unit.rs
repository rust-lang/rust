//@ check-pass
#![deny(improper_c_fn_definitions)]

// Issue: https://github.com/rust-lang/rust/issues/113436
// `()` in (fnptr!) return types and ADT fields should be safe

#[repr(C)]
pub struct Wrap<T>(T);

#[repr(transparent)]
pub struct TransparentWrap<T>(T);

pub extern "C" fn f() -> Wrap<()> {
    todo!()
}

const _: extern "C" fn() -> Wrap<()> = f;

pub extern "C" fn ff() -> Wrap<Wrap<()>> {
    todo!()
}

const _: extern "C" fn() -> Wrap<Wrap<()>> = ff;

pub extern "C" fn g() -> TransparentWrap<()> {
    todo!()
}

const _: extern "C" fn() -> TransparentWrap<()> = g;

pub extern "C" fn gg() -> TransparentWrap<TransparentWrap<()>> {
    todo!()
}

const _: extern "C" fn() -> TransparentWrap<TransparentWrap<()>> = gg;

fn main() {}
