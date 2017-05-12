// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// for-loops are expanded in the front end, and use an `iter` ident in their expansion. Check that
// `iter` is not accessible inside the for loop.

fn main() {
    for _ in 0..10 {
        iter.next();  //~ ERROR cannot find value `iter` in this scope
    }
}
