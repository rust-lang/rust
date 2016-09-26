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
//         StorageLive(var0);               // scope 0 at storage_ranges.rs:14:9: 14:10
//         var0 = const 0i32;               // scope 0 at storage_ranges.rs:14:13: 14:14
//         StorageLive(var1);               // scope 1 at storage_ranges.rs:16:13: 16:14
//         StorageLive(tmp1);               // scope 1 at storage_ranges.rs:16:18: 16:25
//         StorageLive(tmp2);               // scope 1 at storage_ranges.rs:16:23: 16:24
//         tmp2 = var0;                     // scope 1 at storage_ranges.rs:16:23: 16:24
//         tmp1 = std::option::Option<i32>::Some(tmp2,); // scope 1 at storage_ranges.rs:16:18: 16:25
//         var1 = &tmp1;                    // scope 1 at storage_ranges.rs:16:17: 16:25
//         StorageDead(tmp2);               // scope 1 at storage_ranges.rs:16:23: 16:24
//         tmp0 = ();                       // scope 2 at storage_ranges.rs:15:5: 17:6
//         StorageDead(tmp1);               // scope 1 at storage_ranges.rs:16:18: 16:25
//         StorageDead(var1);               // scope 1 at storage_ranges.rs:16:13: 16:14
//         StorageLive(var2);               // scope 1 at storage_ranges.rs:18:9: 18:10
//         var2 = const 1i32;               // scope 1 at storage_ranges.rs:18:13: 18:14
//         return = ();                     // scope 3 at storage_ranges.rs:13:11: 19:2
//         StorageDead(var2);               // scope 1 at storage_ranges.rs:18:9: 18:10
//         StorageDead(var0);               // scope 0 at storage_ranges.rs:14:9: 14:10
//         goto -> bb1;                     // scope 0 at storage_ranges.rs:13:1: 19:2
//     }
//
//     bb1: {
//         return;                          // scope 0 at storage_ranges.rs:13:1: 19:2
//     }
// END rustc.node4.TypeckMir.before.mir
