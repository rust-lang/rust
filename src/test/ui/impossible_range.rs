// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Make sure that invalid ranges generate an error during HIR lowering, not an ICE

#![feature(inclusive_range_syntax)]

pub fn main() {
    ..;
    0..;
    ..1;
    0..1;
    ..=; //~ERROR inclusive range with no end
         //~^HELP bounded at the end
}

fn _foo1() {
    ..=1;
    0..=1;
    0..=; //~ERROR inclusive range with no end
          //~^HELP bounded at the end
}
