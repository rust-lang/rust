// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A test to ensure that helpful `note` messages aren't emitted more often
//! than necessary.

// Although there are three errors, we should only get two "lint level defined
// here" notes pointing at the `warnings` span, one for each error type.
#![deny(warnings)]

fn main() {
    let theTwo = 2;
    let theOtherTwo = 2;
    println!("{}", theTwo);
}
