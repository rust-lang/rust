// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test to ensure we only report an error for the first issued loan that
// conflicts with a new loan, as opposed to every issued loan.  This keeps us
// down to O(n) errors (for n problem lines), instead of O(n^2) errors.

fn main() {
    let mut x = 1;
    let mut addr;
    loop {
        match 1 {
            1 => { addr = &mut x; }
            //~^ ERROR cannot borrow `x` as mutable more than once at a time
            2 => { addr = &mut x; }
            //~^ ERROR cannot borrow `x` as mutable more than once at a time
            _ => { addr = &mut x; }
            //~^ ERROR cannot borrow `x` as mutable more than once at a time
        }
    }
}
