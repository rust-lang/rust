// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// After the work to reoptimize structs, it became possible for immediate logic to fail.
// This test verifies that it actually works.

fn main() {
    let c = |a: u8, b: u16, c: u8| {
        assert_eq!(a, 1);
        assert_eq!(b, 2);
        assert_eq!(c, 3);
    };
    c(1, 2, 3);
}
