// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    // -------- Simplified test case --------

    let _ = || 0..=1;

    // -------- Original test case --------

    let full_length = 1024;
    let range = {
        // do some stuff, omit here
        None
    };

    let range = range.map(|(s, t)| s..=t).unwrap_or(0..=(full_length-1));

    assert_eq!(range, 0..=1023);
}
