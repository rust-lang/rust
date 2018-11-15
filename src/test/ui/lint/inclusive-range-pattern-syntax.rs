// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-pass
// run-rustfix

#![warn(ellipsis_inclusive_range_patterns)]

fn main() {
    let despondency = 2;
    match despondency {
        1...2 => {}
        //~^ WARN `...` range patterns are deprecated
        _ => {}
    }

    match &despondency {
        &1...2 => {}
        //~^ WARN `...` range patterns are deprecated
        _ => {}
    }
}
