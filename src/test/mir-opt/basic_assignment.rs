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
// START rustc.node4.SimplifyCfg-initial.after.mir
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
//         StorageLive(_5);
//         StorageLive(_6);
//         _6 = _4;
//         replace(_5 <- _6) -> [return: bb1, unwind: bb7];
//     }
//     bb1: {
//         drop(_6) -> [return: bb8, unwind: bb5];
//     }
//     bb2: {
//         resume;
//     }
//     bb3: {
//         drop(_4) -> bb2;
//     }
//     bb4: {
//         goto -> bb3;
//     }
//     bb5: {
//         drop(_5) -> bb4;
//     }
//     bb6: {
//         goto -> bb5;
//     }
//     bb7: {
//         drop(_6) -> bb6;
//     }
//     bb8: {
//         StorageDead(_6);
//         _0 = ();
//         drop(_5) -> [return: bb9, unwind: bb3];
//     }
//     bb9: {
//         StorageDead(_5);
//         drop(_4) -> bb10;
//     }
//     bb10: {
//         StorageDead(_4);
//         StorageDead(_2);
//         StorageDead(_1);
//         return;
//     }
// END rustc.node4.SimplifyCfg-initial.after.mir
