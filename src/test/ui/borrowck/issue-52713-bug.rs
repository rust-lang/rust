// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for a bug in #52713: this was an optimization for
// computing liveness that wound up accidentally causing the program
// below to be accepted.

#![feature(nll)]

fn foo<'a>(x: &'a mut u32) -> u32 {
    let mut x = 22;
    let y = &x;
    if false {
        return x;
    }

    x += 1; //~ ERROR
    println!("{}", y);
    return 0;
}

fn main() { }
