// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    if false {
        println!("hello world!");
    }
}

// END RUST SOURCE
// START rustc.node4.SimplifyBranches.initial-before.mir
// bb0: {
//     if(const false) -> [true: bb1, false: bb2];
// }
// END rustc.node4.SimplifyBranches.initial-before.mir
// START rustc.node4.SimplifyBranches.initial-after.mir
// bb0: {
//     goto -> bb2;
// }
// END rustc.node4.SimplifyBranches.initial-after.mir
