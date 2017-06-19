// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z identify_regions
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
// START rustc.node4.SimplifyCfg-qualify-consts.after.mir
//     let mut _0: ();
//     let _1: D;
//     let _3: i32;
//     let _4: &'6_2rce i32;
//     let _7: &'6_4rce i32;
//     let mut _5: ();
//     let mut _6: i32;
//
//     bb0: {
//         StorageLive(_1);
//         _1 = D::{{constructor}}(const 0i32,);
//         StorageLive(_3);
//         _3 = const 0i32;
//         StorageLive(_4);
//         _4 = &'6_2rce _3;
//         StorageLive(_6);
//         _6 = (*_4);
//         _5 = const foo(_6) -> [return: bb2, unwind: bb3];
//     }
//     bb1: {
//         resume;
//     }
//     bb2: {
//         StorageDead(_6);
//         StorageLive(_7);
//         _7 = &'6_4rce _3;
//         _0 = ();
//         StorageDead(_7);
//         EndRegion('6_4rce);
//         StorageDead(_4);
//         EndRegion('6_2rce);
//         StorageDead(_3);
//         drop(_1) -> bb4;
//     }
//     bb3: {
//         EndRegion('6_2rce);
//         drop(_1) -> bb1;
//     }
//     bb4: {
//         StorageDead(_1);
//         return;
//     }
// END rustc.node4.SimplifyCfg-qualify-consts.after.mir
