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
    let q = p;
    if true {
        use_x(*q);
    } else {
        use_x(22);
    }
}

// END RUST SOURCE
// START rustc.main.nll.0.mir
// | '_#2r    | {bb2[0..=6], bb3[0..=1]}
// | '_#3r    | {bb2[1..=6], bb3[0..=1]}
// | '_#4r    | {bb2[5..=6], bb3[0..=1]}
// END rustc.main.nll.0.mir
// START rustc.main.nll.0.mir
// let _2: &'_#3r usize;
// ...
// let _6: &'_#4r usize;
// ...
// _2 = &'_#2r _1[_3];
// ...
// _7 = _2;
// ...
// _6 = move _7;
// END rustc.main.nll.0.mir
