#![crate_type="lib"]
#![no_std]
#![feature(c_variadic)]

use core::ffi::VaList;

pub unsafe extern "C" fn no_escape0<'a>(_: usize, ap: ...) -> VaList<'a> {
    ap //~ ERROR: explicit lifetime required
}

pub unsafe extern "C" fn no_escape1(_: usize, ap: ...) -> VaList<'static> {
    ap //~ ERROR: explicit lifetime required
}

pub unsafe extern "C" fn no_escape2(_: usize, ap: ...) {
    let _ = ap.copy(|ap| { ap }); //~ ERROR: cannot infer an appropriate lifetime
}

pub unsafe extern "C" fn no_escape3(_: usize, mut ap0: &mut VaList, mut ap1: ...) {
    *ap0 = ap1; //~ ERROR: mismatched types
}

pub unsafe extern "C" fn no_escape4(_: usize, ap0: &mut VaList, mut ap1: ...) {
    ap0 = &mut ap1;
    //~^ ERROR: a value of type `core::ffi::VaList<'_>` is borrowed for too long
    //~^^ ERROR: mismatched types
    //~^^^ ERROR: mismatched types
    //~^^^^ ERROR: cannot infer an appropriate lifetime
}
