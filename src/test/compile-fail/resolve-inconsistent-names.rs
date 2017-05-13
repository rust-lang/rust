// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    let y = 1;
    match y {
       a | b => {} //~  ERROR variable `a` from pattern #1 is not bound in pattern #2
                   //~^ ERROR variable `b` from pattern #2 is not bound in pattern #1
                   //~| NOTE pattern doesn't bind `a`
                   //~| NOTE pattern doesn't bind `b`
    }
}
