//@ build-fail
//@ compile-flags: --crate-type=lib
#![allow(internal_features)]
#![feature(rustc_attrs)]
#![feature(required_inlining)]

// Test that required inlining w/ warnings works as expected.

#[rustc_no_mir_inline]
#[inline(must)]
pub fn callee() {
}

#[rustc_no_mir_inline]
#[inline(must("maintain performance characteristics"))]
pub fn callee_justified() {
}

pub fn caller() {
    callee();
//~^ WARN `callee` could not be inlined into `caller` but must be inlined

    callee_justified();
//~^ WARN `callee_justified` could not be inlined into `caller` but must be inlined
}

#[deny(must_inline)]
pub fn caller_overridden() {
    callee();
//~^ ERROR `callee` could not be inlined into `caller_overridden` but must be inlined

    callee_justified();
//~^ ERROR `callee_justified` could not be inlined into `caller_overridden` but must be inlined
}
