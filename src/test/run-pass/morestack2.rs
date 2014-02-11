// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-test newsched transition

fn getbig(i: int) -> int {
    let m = if i >= 0 {
        let j = getbig(i - 1);
        let k = getbig(j - 1);
        k
    } else {
        0
    };
    m
}

pub fn main() {
    getbig(10000);
}
