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
// START rustc.main.nll.0.mir
// | '_#2r    | {bb2[0..=1], bb3[0..=1]}
// | '_#3r    | {bb2[1], bb3[0..=1]}
// ...
//             let _2: &'_#3r usize;
// END rustc.main.nll.0.mir
// START rustc.main.nll.0.mir
//    bb2: {
//            | Live variables on entry to bb2[0]: [_1, _3]
//        _2 = &'_#2r _1[_3];
//            | Live variables on entry to bb2[1]: [_2]
//        switchInt(const true) -> [false: bb4, otherwise: bb3];
//    }
// END rustc.main.nll.0.mir
// START rustc.main.nll.0.mir
//    bb3: {
//            | Live variables on entry to bb3[0]: [_2]
//        StorageLive(_7);
//            | Live variables on entry to bb3[1]: [_2]
//        _7 = (*_2);
//            | Live variables on entry to bb3[2]: [_7]
//        _6 = const use_x(move _7) -> [return: bb5, unwind: bb1];
//    }
// END rustc.main.nll.0.mir
