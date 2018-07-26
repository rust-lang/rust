// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z identify_regions -Z emit-end-regions
// ignore-tidy-linelength

// We will EndRegion for borrows in a loop that occur before break but
// not those after break.

fn main() {
    loop {
        let a = true;
        let b = &a;
        if a { break; }
        let c = &a;
    }
}

// END RUST SOURCE
// START rustc.main.SimplifyCfg-qualify-consts.after.mir
//     let mut _0: ();
//     ...
//     let _7: &'23_3rs bool;
//     ...
//     let _3: &'23_1rs bool;
//     ...
//     let _2: bool;
//     ...
//     let mut _4: ();
//     let mut _5: bool;
//     ...
//     bb0: {
//         goto -> bb1;
//     }
//     bb1: {
//          falseUnwind -> [real: bb2, cleanup: bb3];
//     }
//     bb2: {
//         StorageLive(_2);
//         _2 = const true;
//         StorageLive(_3);
//         _3 = &'23_1rs _2;
//         StorageLive(_5);
//         _5 = _2;
//         switchInt(move _5) -> [false: bb5, otherwise: bb4];
//     }
//     bb3: {
//         ...
//     }
//     bb4: {
//         _0 = ();
//         StorageDead(_5);
//         EndRegion('23_1rs);
//         StorageDead(_3);
//         StorageDead(_2);
//         return;
//     }
//     bb5: {
//         _4 = ();
//         StorageDead(_5);
//         StorageLive(_7);
//         _7 = &'23_3rs _2;
//         _1 = ();
//         EndRegion('23_3rs);
//         StorageDead(_7);
//         EndRegion('23_1rs);
//         StorageDead(_3);
//         StorageDead(_2);
//         goto -> bb1;
//     }
// END rustc.main.SimplifyCfg-qualify-consts.after.mir
