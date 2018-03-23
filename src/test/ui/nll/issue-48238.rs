// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for issue #48238

#![feature(nll)]

fn use_val<'a>(val: &'a u8) -> &'a u8 {
    val
}

fn main() {
    let orig: u8 = 5;
    move || use_val(&orig); //~ ERROR free region `` does not outlive free region `'_#1r`
}
