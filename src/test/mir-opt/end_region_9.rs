// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z identify_regions -Z span_free_formats -Z emit-end-regions
// ignore-tidy-linelength

// This test models a scenario that arielb1 found during review.
// Namely, any filtering of EndRegions must ensure to continue to emit
// any necessary EndRegions that occur earlier in the source than the
// first borrow involving that region.
//
// It is tricky to actually construct examples of this, which is the
// main reason that I am keeping this test even though I have now
// removed the pre-filter that motivated the test in the first place.

fn main() {
    let mut second_iter = false;
    let x = 3;
    'a: loop {
        let mut y;
        loop {
            if second_iter {
                break 'a; // want to generate `EndRegion('a)` here
            } else {
                y = &/*'a*/ x;
            }
            second_iter = true;
        }
    }
}

// END RUST SOURCE
// START rustc.main.SimplifyCfg-qualify-consts.after.mir
// fn main() -> () {
//     let mut _0: ();
//     ...
//     let mut _1: bool;
//     ...
//     let _2: i32;
//     ...
//     let mut _4: &'33_0rs i32;
//     ...
//     let mut _3: ();
//     let mut _5: !;
//     let mut _6: ();
//     let mut _7: bool;
//     let mut _8: !;
//     bb0: {
//        StorageLive(_1);
//        _1 = const false;
//        StorageLive(_2);
//        _2 = const 3i32;
//        StorageLive(_4);
//        goto -> bb1;
//    }
//
//    bb1: {
//        StorageLive(_7);
//        _7 = _1;
//        switchInt(move _7) -> [0u8: bb3, otherwise: bb2];
//    }
//    bb2: {
//        _0 = ();
//        StorageDead(_7);
//        EndRegion('33_0rs);
//        StorageDead(_4);
//        StorageDead(_2);
//        StorageDead(_1);
//        return;
//    }
//    bb3: {
//        _4 = &'33_0rs _2;
//        _6 = ();
//        StorageDead(_7);
//        _1 = const true;
//        _3 = ();
//        goto -> bb1;
//    }
// }
// END rustc.main.SimplifyCfg-qualify-consts.after.mir
