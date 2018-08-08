// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]
#![allow(warnings)]

pub fn fail(x: Option<&(Iterator<Item=()>+Send)>)
            -> Option<&Iterator<Item=()>> {
    // This call used to trigger an LLVM assertion because the return
    // slot had type "Option<&Iterator>"* instead of
    // "Option<&(Iterator+Send)>"* -- but this now yields a
    // compilation error and I'm not sure how to create a comparable
    // test. To ensure that this PARTICULAR failure doesn't occur
    // again, though, I've left this test here, so if this ever starts
    // to compile again, we can adjust the test appropriately (clearly
    // it should never ICE...). -nmatsakis
    inner(x) //~ ERROR mismatched types
}

pub fn inner(x: Option<&(Iterator<Item=()>+Send)>)
             -> Option<&(Iterator<Item=()>+Send)> {
    x
}

#[rustc_error]
fn main() {}
