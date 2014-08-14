// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:macro_crate_test.rs
// ignore-stage1
// ignore-android
// error-pattern: unknown start of token: \x00

// Issue #15750 and #15962 : this test is checking that the standard
// parser rejects embedded idents.  pnkfelix did not want to attempt
// to make a test file that itself used the embedded ident input form,
// since he worrid that would be difficult to work with in many text
// editors, so instead he made a macro that expands into the embedded
// ident form.

#![feature(phase)]

#[phase(plugin)]
extern crate macro_crate_test;

fn main() {
    let x = 0;
    assert_eq!(3, forged_ident!());
}
