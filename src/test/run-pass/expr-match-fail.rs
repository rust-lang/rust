// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn test_simple() {
    let r = match true { true => { true } false => { fail2!() } };
    assert_eq!(r, true);
}

fn test_box() {
    let r = match true { true => { ~[10] } false => { fail2!() } };
    assert_eq!(r[0], 10);
}

pub fn main() { test_simple(); test_box(); }
