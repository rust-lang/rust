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
// START rustc.node4.PreTrans.after.mir
//     bb0: {
//         StorageLive(var0);               // scope 0 at storage_ranges.rs:12:9: 12:10
//         var0 = const 0i32;               // scope 0 at storage_ranges.rs:12:13: 12:14
//         StorageLive(var1);               // scope 1 at storage_ranges.rs:14:13: 14:14
//         StorageLive(tmp1);               // scope 1 at storage_ranges.rs:14:18: 14:25
//         StorageLive(tmp2);               // scope 1 at storage_ranges.rs:14:23: 14:24
//         tmp2 = var0;                     // scope 1 at storage_ranges.rs:14:23: 14:24
//         tmp1 = std::option::Option<i32>::Some(tmp2,); // scope 1 at storage_ranges.rs:14:18: 14:25
//         var1 = &tmp1;                    // scope 1 at storage_ranges.rs:14:17: 14:25
//         StorageDead(tmp2);               // scope 1 at storage_ranges.rs:14:23: 14:24
//         tmp0 = ();                       // scope 2 at storage_ranges.rs:13:5: 15:6
//         StorageDead(tmp1);               // scope 1 at storage_ranges.rs:14:18: 14:25
//         StorageDead(var1);               // scope 1 at storage_ranges.rs:14:13: 14:14
//         StorageLive(var2);               // scope 1 at storage_ranges.rs:16:9: 16:10
//         var2 = const 1i32;               // scope 1 at storage_ranges.rs:16:13: 16:14
//         return = ();                     // scope 3 at storage_ranges.rs:11:11: 17:2
//         StorageDead(var2);               // scope 1 at storage_ranges.rs:16:9: 16:10
//         StorageDead(var0);               // scope 0 at storage_ranges.rs:12:9: 12:10
//         goto -> bb1;                     // scope 0 at storage_ranges.rs:11:1: 17:2
//     }
//
//     bb1: {
//         return;                          // scope 0 at storage_ranges.rs:11:1: 17:2
//     }
// END rustc.node4.PreTrans.after.mir
