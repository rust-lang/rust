#![crate_type = "lib"]
#![no_std]
#![feature(c_variadic)]

use core::ffi::{VaList, va_copy};

pub unsafe extern "C" fn no_escape0<'f>(_: usize, ap: ...) -> VaList<'f> {
    ap
    //~^ ERROR: lifetime may not live long enough
    //~| ERROR: lifetime may not live long enough
}

pub unsafe extern "C" fn no_escape1(_: usize, ap: ...) -> VaList<'static> {
    ap //~ ERROR: lifetime may not live long enough
}

pub unsafe extern "C" fn no_escape2(_: usize, ap: ...) {
    let _ = ap.with_copy(|ap| ap); //~ ERROR: lifetime may not live long enough
}

pub unsafe extern "C" fn no_escape3(_: usize, mut ap0: &mut VaList, mut ap1: ...) {
    *ap0 = ap1;
    //~^ ERROR: lifetime may not live long enough
    //~| ERROR: lifetime may not live long enough
}

pub unsafe extern "C" fn no_escape4(_: usize, mut ap0: &mut VaList, mut ap1: ...) {
    ap0 = &mut ap1;
    //~^ ERROR: `ap1` does not live long enough
    //~| ERROR: lifetime may not live long enough
    //~| ERROR: lifetime may not live long enough
}

pub unsafe extern "C" fn no_escape5(_: usize, mut ap0: &mut VaList, mut ap1: ...) {
    *ap0 = va_copy!(ap1);
    //~^ ERROR: lifetime may not live long enough
    //~| ERROR: temporary value dropped while borrowed
}

pub unsafe extern "C" fn no_escape6<'f>(ap: ...) -> VaList<'f> {
    va_copy!(ap)
    //~^ ERROR: lifetime may not live long enough
    //~| ERROR: cannot return value referencing temporary value
}

pub unsafe extern "C" fn no_escape7(ap: ...) -> VaList<'static> {
    va_copy!(ap)
    //~^ ERROR: lifetime may not live long enough
    //~| ERROR: cannot return value referencing temporary value
}
