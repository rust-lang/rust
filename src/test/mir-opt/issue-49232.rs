// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// We must mark a variable whose initialization fails due to an
// abort statement as StorageDead.

fn main() {
    loop {
        let beacon = {
            match true {
                false => 4,
                true => break,
            }
        };
        drop(&beacon);
    }
}

// END RUST SOURCE
// START rustc.main.mir_map.0.mir
// fn main() -> (){
//     let mut _0: ();
//     scope 1 {
//     }
//     scope 2 {
//         let _2: i32;
//     }
//     let mut _1: ();
//     let mut _3: bool;
//     let mut _4: !;
//     let mut _5: ();
//     let mut _6: &i32;
//     bb0: {
//         goto -> bb1;
//     }
//     bb1: {
//         falseUnwind -> [real: bb3, cleanup: bb4];
//     }
//     bb2: {
//         goto -> bb20;
//     }
//     bb3: {
//         StorageLive(_2);
//         StorageLive(_3);
//         _3 = const true;
//         FakeRead(ForMatchedPlace, _3);
//         switchInt(_3) -> [false: bb11, otherwise: bb10];
//     }
//     bb4: {
//         resume;
//     }
//     bb5: {
//         _2 = const 4i32;
//         goto -> bb14;
//     }
//     bb6: {
//         _0 = ();
//         goto -> bb15;
//     }
//     bb7: {
//         falseEdges -> [real: bb12, imaginary: bb8];
//     }
//     bb8: {
//         falseEdges -> [real: bb13, imaginary: bb9];
//     }
//     bb9: {
//         unreachable;
//     }
//     bb10: {
//         goto -> bb8;
//     }
//     bb11: {
//         goto -> bb7;
//     }
//     bb12: {
//         goto -> bb5;
//     }
//     bb13: {
//         goto -> bb6;
//     }
//     bb14: {
//         FakeRead(ForLet, _2);
//         StorageDead(_3);
//         StorageLive(_6);
//         _6 = &_2;
//         _5 = const std::mem::drop(move _6) -> [return: bb19, unwind: bb4];
//     }
//     bb15: {
//         StorageDead(_3);
//         goto -> bb16;
//     }
//     bb16: {
//         StorageDead(_2);
//         goto -> bb2;
//     }
//     bb17: {
//         _4 = ();
//         unreachable;
//     }
//     bb18: {
//         StorageDead(_4);
//         goto -> bb14;
//     }
//     bb19: {
//         StorageDead(_6);
//         _1 = ();
//         StorageDead(_2);
//         goto -> bb1;
//     }
//     bb20: {
//         return;
//     }
// }
// END rustc.main.mir_map.0.mir
