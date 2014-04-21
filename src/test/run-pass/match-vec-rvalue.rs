// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that matching rvalues with drops does not crash.


pub fn main() {
    match vec!(1i, 2i, 3i) {
        x => {
            assert_eq!(x.len(), 3);
            assert_eq!(*x.get(0), 1);
            assert_eq!(*x.get(1), 2);
            assert_eq!(*x.get(2), 3);
        }
    }
}
