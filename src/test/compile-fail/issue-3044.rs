// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern: Non-function passed to a `do` function as its last argument, or wrong number of arguments passed to a `do` function
fn main() {
    let needlesArr: ~[char] = ~['a', 'f'];
    do vec::foldr(needlesArr) |x, y| {
    }
// for some reason if I use the new error syntax for the two error messages this generates,
// the test runner gets confused -- tjc
}

