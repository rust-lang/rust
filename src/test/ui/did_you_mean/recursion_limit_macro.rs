// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that the recursion limit can be changed and that the compiler
// suggests a fix. In this case, we have a recursing macro that will
// overflow if the number of arguments surpasses the recursion limit.

#![allow(dead_code)]
#![recursion_limit="10"]

macro_rules! recurse {
    () => { };
    ($t:tt $($tail:tt)*) => { recurse!($($tail)*) }; //~ ERROR recursion limit
}

fn main() {
    recurse!(0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9);
}

