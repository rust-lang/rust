// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-compare-mode-nll

// Check that E0161 is a hard error in all possible configurations that might
// affect it.

// revisions: ast nll zflags edition astul nllul zflagsul editionul
//[zflags]compile-flags: -Z borrowck=migrate -Z two-phase-borrows
//[edition]edition:2018
//[zflagsul]compile-flags: -Z borrowck=migrate -Z two-phase-borrows
//[editionul]edition:2018

#![cfg_attr(nll, feature(nll))]
#![cfg_attr(nllul, feature(nll))]
#![cfg_attr(astul, feature(unsized_locals))]
#![cfg_attr(zflagsul, feature(unsized_locals))]
#![cfg_attr(nllul, feature(unsized_locals))]
#![cfg_attr(editionul, feature(unsized_locals))]

#![feature(box_syntax)]

fn foo(x: Box<[i32]>) {
    box *x;
    //[ast]~^ ERROR E0161
    //[nll]~^^ ERROR E0161
    //[zflags]~^^^ ERROR E0161
    //[edition]~^^^^ ERROR E0161
    //[astul]~^^^^^ ERROR E0161
    //[nllul]~^^^^^^ ERROR E0161
    //[zflagsul]~^^^^^^^ ERROR E0161
    //[editionul]~^^^^^^^^ ERROR E0161
}

fn main() {}
