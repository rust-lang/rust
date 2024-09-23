//@ build-fail
//@ compile-flags: --crate-type=lib
#![allow(internal_features)]
#![feature(rustc_attrs)]
#![feature(required_inlining)]

// Test that required inlining w/ errors works as expected.

#[rustc_no_mir_inline]
#[inline(required)]
pub fn callee() {
}

#[rustc_no_mir_inline]
#[inline(required("maintain security properties"))]
pub fn callee_justified() {
}

pub fn caller() {
    callee();
//~^ ERROR `callee` could not be inlined into `caller` but is required to be inlined

    callee_justified();
//~^ ERROR `callee_justified` could not be inlined into `caller` but is required to be inlined
}

#[warn(required_inline)]
pub fn caller_overridden() {
    callee();
//~^ WARN `callee` could not be inlined into `caller_overridden` but is required to be inlined

    callee_justified();
//~^ WARN `callee_justified` could not be inlined into `caller_overridden` but is required to be inlined
}
