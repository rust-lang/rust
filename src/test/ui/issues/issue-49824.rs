// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]

// This test checks that a failure occurs with NLL but does not fail with the
// legacy AST output. Check issue-49824.nll.stderr for expected compilation error
// output under NLL and #49824 for more information.
//
// UPDATE: Once #55756 was fixed, this test started to fail in both modes.

#[rustc_error]
fn main() {
    let mut x = 0;
    || {
        || { //~ ERROR too short
            let _y = &mut x;
        }
    };
}
