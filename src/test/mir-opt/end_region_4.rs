// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z identify_regions -Z emit-end-regions
// ignore-tidy-linelength

// Unwinding should EndRegion for in-scope borrows: Direct borrows.

fn main() {
    let d = D(0);
    let a = 0;
    let b = &a;
    foo(*b);
    let c = &a;
}

struct D(i32);
impl Drop for D { fn drop(&mut self) { println!("dropping D({})", self.0); } }

fn foo(i: i32) {
    if i > 0 { panic!("im positive"); }
}

// END RUST SOURCE
// START rustc.main.SimplifyCfg-qualify-consts.after.mir
//     let mut _0: ();
//     ...
//     let _6: &'26_4rs i32;
//     ...
//     let _3: &'26_2rs i32;
//     ...
//     let _2: i32;
//     ...
//     let _1: D;
//     ...
//     let mut _4: ();
//     let mut _5: i32;
//     bb0: {
//         StorageLive(_1);
//         _1 = D::{{constructor}}(const 0i32,);
//         StorageLive(_2);
//         _2 = const 0i32;
//         StorageLive(_3);
//         _3 = &'26_2rs _2;
//         StorageLive(_5);
//         _5 = (*_3);
//         _4 = const foo(move _5) -> [return: bb2, unwind: bb3];
//     }
//     bb1: {
//         resume;
//     }
//     bb2: {
//         StorageDead(_5);
//         StorageLive(_6);
//         _6 = &'26_4rs _2;
//         _0 = ();
//         EndRegion('26_4rs);
//         StorageDead(_6);
//         EndRegion('26_2rs);
//         StorageDead(_3);
//         StorageDead(_2);
//         drop(_1) -> [return: bb4, unwind: bb1];
//     }
//     bb3: {
//         EndRegion('26_2rs);
//         drop(_1) -> bb1;
//     }
//     bb4: {
//         StorageDead(_1);
//         return;
//     }
// END rustc.main.SimplifyCfg-qualify-consts.after.mir
