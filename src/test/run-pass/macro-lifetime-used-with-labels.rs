// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(unreachable_code)]
#![feature(macro_lifetime_matcher)]

macro_rules! x {
    ($a:lifetime) => {
        $a: loop {
            break $a;
            panic!("failed");
        }
    }
}
macro_rules! br {
    ($a:lifetime) => {
        break $a;
    }
}
macro_rules! br2 {
    ($b:lifetime) => {
        'b: loop {
            break $b; // this $b should refer to the outer loop.
        }
    }
}
fn main() {
    x!('a);
    'c: loop {
        br!('c);
        panic!("failed");
    }
    'b: loop {
        br2!('b);
        panic!("failed");
    }
}
