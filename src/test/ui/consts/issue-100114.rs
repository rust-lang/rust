// check-pass

#![allow(warnings)]
#![feature(never_type)]
#![allow(const_err)]

use std::mem::MaybeUninit;

const fn never() -> ! {
    unsafe { MaybeUninit::uninit().assume_init() }
}

const NEVER: ! = never();

fn main() {}
