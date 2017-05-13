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
//         StorageLive(_1);
//         _1 = const 0i32;
//         StorageLive(_3);
//         StorageLive(_4);
//         StorageLive(_5);
//         _5 = _1;
//         _4 = std::option::Option<i32>::Some(_5,);
//         _3 = &_4;
//         StorageDead(_5);
//         _2 = ();
//         StorageDead(_4);
//         StorageDead(_3);
//         StorageLive(_6);
//         _6 = const 1i32;
//         _0 = ();
//         StorageDead(_6);
//         StorageDead(_1);
//         return;
//     }
// END rustc.node4.TypeckMir.before.mir
