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

// TODO The StorageDead for local1 (a) after local6's (c) is missing!

// END RUST SOURCE
// START rustc.node4.TypeckMir.before.mir
//     bb0: {
//         StorageLive(local1);             // scope 0 at storage_ranges.rs:12:9: 12:10
//         local1 = const 0i32;             // scope 0 at storage_ranges.rs:12:13: 12:14
//         StorageLive(local3);             // scope 1 at storage_ranges.rs:14:13: 14:14
//         StorageLive(local4);             // scope 1 at storage_ranges.rs:14:18: 14:25
//         StorageLive(local5);             // scope 1 at storage_ranges.rs:14:23: 14:24
//         local5 = local1;                 // scope 1 at storage_ranges.rs:14:23: 14:24
//         local4 = std::option::Option<i32>::Some(local5,); // scope 1 at storage_ranges.rs:14:18: 14:25
//         local3 = &local4;                // scope 1 at storage_ranges.rs:14:17: 14:25
//         StorageDead(local5);             // scope 1 at storage_ranges.rs:14:23: 14:24
//         local2 = ();                     // scope 2 at storage_ranges.rs:13:5: 15:6
//         StorageDead(local4);             // scope 1 at storage_ranges.rs:14:18: 14:25
//         StorageDead(local3);             // scope 1 at storage_ranges.rs:14:13: 14:14
//         StorageLive(local6);             // scope 1 at storage_ranges.rs:16:9: 16:10
//         local6 = const 1i32;             // scope 1 at storage_ranges.rs:16:13: 16:14
//         local0 = ();                     // scope 3 at storage_ranges.rs:11:11: 17:2
//         StorageDead(local6);             // scope 1 at storage_ranges.rs:16:9: 16:10
//         goto -> bb1;                     // scope 0 at storage_ranges.rs:11:1: 17:2
//     }
//
//     bb1: {
//         return;                          // scope 0 at storage_ranges.rs:13:1: 19:2
//     }
// END rustc.node4.TypeckMir.before.mir
