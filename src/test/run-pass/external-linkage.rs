// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:external_linkage.rs
// xfail-fast

extern mod external_linkage;

#[no_mangle]
pub extern "C" fn foreign() {
}

fn notvisible() {
}

#[no_mangle]
pub fn visible() {
}

#[no_mangle]
pub static x: int = 5;
static y: int = 0;

pub fn main() {
    // this variant of external_linkage will cause the runtime linker to try
    // and find visible() and x, which are public.
    unsafe { external_linkage::doer() };
    notvisible();
    visible();
}
