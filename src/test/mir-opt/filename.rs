// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that we can write tests for lines that contain the filename of the test

fn main() {
    let closure = |x: i32| x+x;
    closure(21);
}

// END RUST SOURCE
// START rustc.node15.EraseRegions.after.mir
// fn main::{{closure}}(_1: &[closure@$FILE:14:19: 14:31], _2: i32) -> i32 {
// }
// END rustc.node15.EraseRegions.after.mir
