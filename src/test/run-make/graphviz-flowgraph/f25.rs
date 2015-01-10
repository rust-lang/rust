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
pub fn expr_while_25() {
    let mut x = 25is;
    let mut y = 25is;
    let mut z = 25is;

    'a: loop {
        if x == 0is { break; "unreachable"; }
        x -= 1is;

        'a: loop {
            if y == 0is { break; "unreachable"; }
            y -= 1is;

            'a: loop {
                if z == 0is { break; "unreachable"; }
                z -= 1is;
            }

            if x > 10is {
                continue 'a;
                "unreachable";
            }
        }
    }
}
