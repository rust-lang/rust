// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[deny(unreachable_code)];
#[allow(unused_variable)];

fn fail_len(v: ~[int]) -> uint {
    let mut i = 3;
    fail2!();
    for x in v.iter() { i += 1u; }
    //~^ ERROR: unreachable statement
    return i;
}
fn main() {}
