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
    let mut x = 25i;
    let mut y = 25i;
    let mut z = 25i;

    'a: loop {
        if x == 0i { break; "unreachable"; }
        x -= 1i;

        'a: loop {
            if y == 0i { break; "unreachable"; }
            y -= 1i;

            'a: loop {
                if z == 0i { break; "unreachable"; }
                z -= 1i;
            }

            if x > 10i {
                continue 'a;
                "unreachable";
            }
        }
    }
}
