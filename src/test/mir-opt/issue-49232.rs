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
//     let mut _4: u8;
//     let mut _5: !;
//     let mut _6: ();
//     let mut _7: &i32;
//     bb0: {
//         goto -> bb1;
//     }
//     bb1: {
//         falseUnwind -> [real: bb3, cleanup: bb4];
//     }
//     bb2: {
//         goto -> bb29;
//     }
//     bb3: {
//         StorageLive(_2);
//         StorageLive(_3);
//         _3 = const true;
//         _4 = discriminant(_3);
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
//         StorageDead(_3);
//         StorageLive(_7);
//         _7 = &_2;
//         _6 = const std::mem::drop(move _7) -> [return: bb28, unwind: bb4];
//     }
//     bb15: {
//         goto -> bb16;
//     }
//     bb16: {
//         goto -> bb17;
//     }
//     bb17: {
//         goto -> bb18;
//     }
//     bb18: {
//         goto -> bb19;
//     }
//     bb19: {
//         goto -> bb20;
//     }
//     bb20: {
//         StorageDead(_3);
//         goto -> bb21;
//     }
//     bb21: {
//         goto -> bb22;
//     }
//     bb22: {
//         StorageDead(_2);
//         goto -> bb23;
//     }
//     bb23: {
//         goto -> bb24;
//     }
//     bb24: {
//         goto -> bb25;
//     }
//     bb25: {
//         goto -> bb2;
//     }
//     bb26: {
//         _5 = ();
//         unreachable;
//     }
//     bb27: {
//         StorageDead(_5);
//         goto -> bb14;
//     }
//     bb28: {
//         StorageDead(_7);
//         _1 = ();
//         StorageDead(_2);
//         goto -> bb1;
//     }
//     bb29: {
//         return;
//     }
// }
// END rustc.main.mir_map.0.mir
