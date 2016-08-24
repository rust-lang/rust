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
    let x: (isize, &[isize]) = (2, &[1, 2]);
    assert_eq!(match x {
        (0, &[_, _]) => 0,
        (1, _) => 1,
        (2, &[_, _]) => 2,
        (2, _) => 3,
        _ => 4
    }, 2);
}
