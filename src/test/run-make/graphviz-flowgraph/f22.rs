// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(unreachable_code)]
pub fn expr_break_label_21() {
    let mut x = 15i;
    let mut y = 151i;
    'outer: loop {
        'inner: loop {
            if x == 1i {
                continue 'outer;
                "unreachable";
            }
            if y >= 2i {
                return;
                "unreachable";
            }
            x -= 1i;
            y -= 3i;
        }
        "unreachable";
    }
    "unreachable";
}
