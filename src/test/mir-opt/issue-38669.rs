// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// check that we don't StorageDead booleans before they are used

fn main() {
    let mut should_break = false;
    loop {
        if should_break {
            break;
        }
        should_break = true;
    }
}

// END RUST SOURCE
// START rustc.main.SimplifyCfg-initial.after.mir
//     bb0: {
//         StorageLive(_1);
//         _1 = const false;
//         goto -> bb2;
//     }
//
//     bb1: {
//         resume;
//     }
//     bb2: {
//         StorageLive(_4);
//         _4 = _1;
//         switchInt(move _4) -> [0u8: bb4, otherwise: bb3];
//     }
//     bb3: {
//         _0 = ();
//         StorageDead(_4);
//         StorageDead(_1);
//         return;
//     }
//
//     bb4: {
//         _3 = ();
//         StorageDead(_4);
//         _1 = const true;
//         _2 = ();
//         falseUnwind -> [real: bb2, cleanup: bb1];
//     }
// END rustc.main.SimplifyCfg-initial.after.mir
