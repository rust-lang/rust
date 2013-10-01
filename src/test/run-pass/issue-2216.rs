// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub fn main() {
    let mut x = 0;

    'foo: loop {
        'bar: loop {
            'quux: loop {
                if 1 == 2 {
                    break 'foo;
                }
                else {
                    break 'bar;
                }
            }
            continue 'foo;
        }
        x = 42;
        break;
    }

    error2!("{:?}", x);
    assert_eq!(x, 42);
}
