// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-pass

// This test is bogus (i.e. should be compile-fail) during the period
// where #54986 is implemented and #54987 is *not* implemented. For
// now: just ignore it under nll
//
// ignore-compare-mode-nll

// This test is checking that the write to `c.0` (which has been moved out of)
// won't overwrite the state in `c2`.
//
// That's a fine thing to test when this code is accepted by the
// compiler, and this code is being transcribed accordingly into
// the ui test issue-21232-partial-init-and-use.rs

fn main() {
    let mut c = (1, "".to_owned());
    match c {
        c2 => {
            c.0 = 2;
            assert_eq!(c2.0, 1);
        }
    }
}
