// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]

fn main() {
    let x = box S::new();
    drop(x);
}

struct S;

impl S {
    fn new() -> Self { S }
}

impl Drop for S {
    fn drop(&mut self) {
        println!("splat!");
    }
}

// END RUST SOURCE
// START rustc.node4.ElaborateDrops.before.mir
//     let mut _0: ();
//     let _1: std::boxed::Box<S>;
//     let mut _2: std::boxed::Box<S>;
//     let mut _3: ();
//     let mut _4: std::boxed::Box<S>;
//
//     bb0: {
//         StorageLive(_1);
//         StorageLive(_2);
//         _2 = Box(S);
//         (*_2) = const S::new() -> [return: bb1, unwind: bb3];
//     }
//
//     bb1: {
//         _1 = _2;
//         drop(_2) -> bb4;
//     }
//
//     bb2: {
//         resume;
//     }
//
//     bb3: {
//         drop(_2) -> bb2;
//     }
//
//     bb4: {
//         StorageDead(_2);
//         StorageLive(_4);
//         _4 = _1;
//         _3 = const std::mem::drop(_4) -> [return: bb5, unwind: bb7];
//     }
//
//     bb5: {
//         drop(_4) -> [return: bb8, unwind: bb6];
//     }
//
//     bb6: {
//         drop(_1) -> bb2;
//     }
//
//     bb7: {
//         drop(_4) -> bb6;
//     }
//
//     bb8: {
//         StorageDead(_4);
//         _0 = ();
//         drop(_1) -> bb9;
//     }
//
//     bb9: {
//         StorageDead(_1);
//         return;
//     }
// }
// END rustc.node4.ElaborateDrops.before.mir
