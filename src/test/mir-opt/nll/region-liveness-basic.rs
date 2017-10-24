// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Basic test for liveness constraints: the region (`R1`) that appears
// in the type of `p` includes the points after `&v[0]` up to (but not
// including) the call to `use_x`. The `else` branch is not included.

// compile-flags:-Znll -Zverbose
//                     ^^^^^^^^^ force compiler to dump more region information

#![allow(warnings)]

fn use_x(_: usize) -> bool { true }

fn main() {
    let mut v = [1, 2, 3];
    let p = &v[0];
    if true {
        use_x(*p);
    } else {
        use_x(22);
    }
}

// END RUST SOURCE
// START rustc.node12.nll.0.mir
// | R0: {bb1[1], bb2[0], bb2[1]}
// | R1: {bb1[1], bb2[0], bb2[1]}
// ...
//             let _2: &'_#1r usize;
// END rustc.node12.nll.0.mir
// START rustc.node12.nll.0.mir
//    bb1: {
//        | Regular-Live variables here: [_1, _3]
//        | Drop-Live variables here: []
//        _2 = &'_#0r _1[_3];
//        | Regular-Live variables here: [_2]
//        | Drop-Live variables here: []
//        switchInt(const true) -> [0u8: bb3, otherwise: bb2];
//    }
// END rustc.node12.nll.0.mir
// START rustc.node12.nll.0.mir
//    bb2: {
//        | Regular-Live variables here: [_2]
//        | Drop-Live variables here: []
//        StorageLive(_7);
//        | Regular-Live variables here: [_2]
//        | Drop-Live variables here: []
//        _7 = (*_2);
//        | Regular-Live variables here: [_7]
//        | Drop-Live variables here: []
//        _6 = const use_x(_7) -> bb4;
//    }
// END rustc.node12.nll.0.mir
