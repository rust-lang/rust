// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(unreachable_code)]
#![allow(unused_variables)]
#![allow(dead_code)]

fn fail_len(v: Vec<isize> ) -> usize {
    let mut i = 3;
    panic!();
    for x in &v { i += 1_usize; }
    //~^ ERROR: unreachable statement
    return i;
}
fn main() {}
