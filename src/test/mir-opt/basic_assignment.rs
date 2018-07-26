// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// this tests move up progration, which is not yet implemented
// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// check that codegen of assignment expressions is sane. Assignments
// tend to be absent in simple code, so subtle breakage in them can
// leave a quite hard-to-find trail of destruction.

// ignore-tidy-linelength

fn main() {
    let nodrop_x = false;
    let nodrop_y;

    nodrop_y = nodrop_x;

    let drop_x : Option<Box<u32>> = None;
    let drop_y;

    drop_y = drop_x;
}

// END RUST SOURCE
// START rustc.main.SimplifyCfg-initial.after.mir
//     bb0: {
//         StorageLive(_1);
//         _1 = const false;
//         StorageLive(_2);
//         StorageLive(_3);
//         _3 = _1;
//         _2 = move _3;
//         StorageDead(_3);
//         StorageLive(_4);
//         UserAssertTy(Canonical { variables: [], value: std::option::Option<std::boxed::Box<u32>> }, _4);
//         _4 = std::option::Option<std::boxed::Box<u32>>::None;
//         StorageLive(_5);
//         StorageLive(_6);
//         _6 = move _4;
//         replace(_5 <-move _6) -> [return: bb2, unwind: bb5];
//     }
//     bb1: {
//         resume;
//     }
//     bb2: {
//         drop(_6) -> [return: bb6, unwind: bb4];
//     }
//     bb3: {
//         drop(_4) -> bb1;
//     }
//     bb4: {
//         drop(_5) -> bb3;
//     }
//     bb5: {
//         drop(_6) -> bb4;
//     }
//     bb6: {
//         StorageDead(_6);
//         _0 = ();
//         drop(_5) -> [return: bb7, unwind: bb3];
//     }
//     bb7: {
//         StorageDead(_5);
//         drop(_4) -> [return: bb8, unwind: bb1];
//     }
//     bb8: {
//         StorageDead(_4);
//         StorageDead(_2);
//         StorageDead(_1);
//         return;
//     }
// END rustc.main.SimplifyCfg-initial.after.mir
