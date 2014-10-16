// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(macro_rules)]

macro_rules! prob1 {
    (0) => {
        0
    };
    ($n:expr) => {
        if ($n % 3 == 0) || ($n % 5 == 0) {
            $n + prob1!($n - 1); //~ ERROR recursion limit reached while expanding the macro `prob1`
        } else {
            prob1!($n - 1);
        }
    };
}

fn main() {
    println!("Problem 1: {}", prob1!(1000));
}
