// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: --edition 2015

#![deny(rust_2018_compatibility)]

// Don't make a suggestion for a raw identifer replacement unless raw
// identifiers are enabled.

fn main() {
    let async = 3; //~ ERROR: is a keyword
    //~^ WARN previously accepted
}
