// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Make sure we drop the refs of the temporaries needed to return the
// values from the else if branch
pub fn main() {
    let y: @uint = @10u;
    let _x = if false { y } else if true { y } else { y };
    assert_eq!(*y, 10u);
}
