// Regression test for #122659
//@ build-pass
//@ compile-flags: -O --crate-type=lib

#![feature(core_intrinsics)]
#![feature(const_eval_select)]

use std::intrinsics::const_eval_select;

#[inline]
pub const fn f() {
    const_eval_select((), g, g)
}

#[inline]
pub const fn g() {
    const_eval_select((), f, f)
}
