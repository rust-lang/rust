// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

fn main() {
    let a = 0;
    {
        let b = &Some(a);
    }
    let c = 1;
}

// END RUST SOURCE
// START rustc.node4.TypeckMir.before.mir
//     bb0: {
//         StorageLive(_1);             // scope 0 at storage_ranges.rs:12:9: 12:10
//         _1 = const 0i32;             // scope 0 at storage_ranges.rs:12:13: 12:14
//         StorageLive(_3);             // scope 1 at storage_ranges.rs:14:13: 14:14
//         StorageLive(_4);             // scope 1 at storage_ranges.rs:14:18: 14:25
//         StorageLive(_5);             // scope 1 at storage_ranges.rs:14:23: 14:24
//         _5 = _1;                 // scope 1 at storage_ranges.rs:14:23: 14:24
//         _4 = std::option::Option<i32>::Some(_5,); // scope 1 at storage_ranges.rs:14:18: 14:25
//         _3 = &_4;                // scope 1 at storage_ranges.rs:14:17: 14:25
//         StorageDead(_5);             // scope 1 at storage_ranges.rs:14:23: 14:24
//         _2 = ();                     // scope 2 at storage_ranges.rs:13:5: 15:6
//         StorageDead(_4);             // scope 1 at storage_ranges.rs:14:18: 14:25
//         StorageDead(_3);             // scope 1 at storage_ranges.rs:14:13: 14:14
//         StorageLive(_6);             // scope 1 at storage_ranges.rs:16:9: 16:10
//         _6 = const 1i32;             // scope 1 at storage_ranges.rs:16:13: 16:14
//         _0 = ();                     // scope 3 at storage_ranges.rs:11:11: 17:2
//         StorageDead(_6);             // scope 1 at storage_ranges.rs:16:9: 16:10
//         StorageDead(_1);             // scope 0 at storage_ranges.rs:14:9: 14:10
//         goto -> bb1;                     // scope 0 at storage_ranges.rs:11:1: 17:2
//     }
//
//     bb1: {
//         return;                          // scope 0 at storage_ranges.rs:13:1: 19:2
//     }
// END rustc.node4.TypeckMir.before.mir
