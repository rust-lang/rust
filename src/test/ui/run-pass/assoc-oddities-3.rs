// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn that_odd_parse(c: bool, n: usize) -> u32 {
    let x = 2;
    let a = [1, 2, 3, 4];
    let b = [5, 6, 7, 7];
    x + if c { a } else { b }[n]
}

fn main() {
    assert_eq!(4, that_odd_parse(true, 1));
    assert_eq!(8, that_odd_parse(false, 1));
}
