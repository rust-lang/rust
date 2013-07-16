// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.



fn pairs(it: &fn((int, int))) {
    let mut i: int = 0;
    let mut j: int = 0;
    while i < 10 { it((i, j)); i += 1; j += i; }
}

pub fn main() {
    let mut i: int = 10;
    let mut j: int = 0;
    do pairs() |p| {
        let (_0, _1) = p;
        info!(_0);
        info!(_1);
        assert_eq!(_0 + 10, i);
        i += 1;
        j = _1;
    };
    assert_eq!(j, 45);
}
