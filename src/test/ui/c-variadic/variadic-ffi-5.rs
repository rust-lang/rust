#![crate_type="lib"]
#![no_std]
#![feature(c_variadic)]
// The tests in this file are similar to that of variadic-ffi-4, but this
// one enables nll.
#![feature(nll)]

use core::ffi::VaList;

pub unsafe extern "C" fn no_escape0<'a>(_: usize, ap: ...) -> VaList<'a> {
    ap //~ ERROR: explicit lifetime required
}

pub unsafe extern "C" fn no_escape1(_: usize, ap: ...) -> VaList<'static> {
    ap //~ ERROR: explicit lifetime required
}

pub unsafe extern "C" fn no_escape2(_: usize, ap: ...) {
    let _ = ap.with_copy(|ap| { ap }); //~ ERROR: lifetime may not live long enough
}

pub unsafe extern "C" fn no_escape3(_: usize, ap0: &mut VaList, mut ap1: ...) {
    *ap0 = ap1; //~ ERROR: lifetime may not live long enough
}

pub unsafe extern "C" fn no_escape4(_: usize, mut ap0: &mut VaList, mut ap1: ...) {
    ap0 = &mut ap1;
    //~^ ERROR: lifetime may not live long enough
    //~^^ ERROR: lifetime may not live long enough
    //~^^^ ERROR: `ap1` does not live long enough
}
