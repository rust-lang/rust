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

fn main() {
    let nodrop_x = false;
    let nodrop_y;

    nodrop_y = nodrop_x;

    let drop_x : Option<Box<u32>> = None;
    let drop_y;

    drop_y = drop_x;
}

// END RUST SOURCE
// START rustc.node4.SimplifyCfg.initial-after.mir
//     bb0: {
//         StorageLive(_1);
//         _1 = const false;
//         StorageLive(_2);
//         StorageLive(_3);
//         _3 = _1;
//         _2 = _3;
//         StorageDead(_3);
//         StorageLive(_4);
//         _4 = std::option::Option<std::boxed::Box<u32>>::None;
//         StorageLive(_6);
//         StorageLive(_7);
//         _7 = _4;
//         replace(_6 <- _7) -> [return: bb5, unwind: bb4];
//     }
//     bb1: {
//         resume;
//     }
//     bb2: {
//         drop(_4) -> bb1;
//     }
//     bb3: {
//         drop(_6) -> bb2;
//     }
//     bb4: {
//         drop(_7) -> bb3;
//     }
//     bb5: {
//         drop(_7) -> [return: bb6, unwind: bb3];
//     }
//     bb6: {
//         StorageDead(_7);
//         _0 = ();
//         drop(_6) -> [return: bb7, unwind: bb2];
//     }
//     bb7: {
//         StorageDead(_6);
//         drop(_4) -> bb8;
//     }
//     bb8: {
//         StorageDead(_4);
//         StorageDead(_2);
//         StorageDead(_1);
//         return;
//     }
// END rustc.node4.SimplifyCfg.initial-after.mir
