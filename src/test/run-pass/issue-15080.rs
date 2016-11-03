// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#![feature(slice_patterns)]

fn main() {
    let mut x: &[_] = &[1, 2, 3, 4];

    let mut result = vec![];
    loop {
        x = match *x {
            [1, n, 3, ref rest..] => {
                result.push(n);
                rest
            }
            [n, ref rest..] => {
                result.push(n);
                rest
            }
            [] =>
                break
        }
    }
    assert_eq!(result, [2, 4]);
}
