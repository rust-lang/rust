// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.




// -*- rust -*-
pub fn main() {
    let x: int = 15;
    let y: int = 5;
    fail_unless!((x / 5 == 3));
    fail_unless!((x / 4 == 3));
    fail_unless!((x / 3 == 5));
    fail_unless!((x / y == 3));
    fail_unless!((15 / y == 3));
    fail_unless!((x % 5 == 0));
    fail_unless!((x % 4 == 3));
    fail_unless!((x % 3 == 0));
    fail_unless!((x % y == 0));
    fail_unless!((15 % y == 0));
}
