// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct S;
// Ensure S is moved, not copied, on assignment.
impl Drop for S { fn drop(&mut self) { } }

// user-defined function "returning" bottom (i.e. no return at all).
fn my_fail() -> ! { loop {} }

pub fn step(f: bool) {
    let mut g = S;
    let mut i = 0i;
    loop
    {
        if i > 10 { break; } else { i += 1; }

        let _g = g;

        if f {
            // re-initialize g, but only before restarting loop.
            g = S;
            continue;
        }

        my_fail();

        // we never get here, so we do not need to re-initialize g.
    }
}

pub fn main() {
    step(true);
}
