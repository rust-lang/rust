// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z identify_regions -Z span_free_formats
// ignore-tidy-linelength

// Unwinding should EndRegion for in-scope borrows: Borrowing via by-ref closure.

fn main() {
    let d = D(0);
    foo(|| -> i32 { d.0 });
}

struct D(i32);
impl Drop for D { fn drop(&mut self) { println!("dropping D({})", self.0); } }

fn foo<F>(f: F) where F: FnOnce() -> i32 {
    if f() > 0 { panic!("im positive"); }
}

// END RUST SOURCE
// START rustc.node4.SimplifyCfg-qualify-consts.after.mir
// fn main() -> () {
//     let mut _0: ();
//     let _1: D;
//     let mut _2: ();
//     let mut _3: [closure@NodeId(18) d:&'14mce D];
//     let mut _4: &'14mce D;
//     bb0: {
//         StorageLive(_1);
//         _1 = D::{{constructor}}(const 0i32,);
//         StorageLive(_3);
//         StorageLive(_4);
//         _4 = &'14mce _1;
//         _3 = [closure@NodeId(18)] { d: _4 };
//         StorageDead(_4);
//         _2 = const foo(_3) -> [return: bb1, unwind: bb3];
//     }
//     bb1: {
//         StorageDead(_3);
//         EndRegion('14mce);
//         _0 = ();
//         drop(_1) -> bb4;
//     }
//     bb2: {
//         resume;
//     }
//     bb3: {
//         EndRegion('14mce);
//         drop(_1) -> bb2;
//     }
//     bb4: {
//         StorageDead(_1);
//         return;
//     }
// }
// END rustc.node4.SimplifyCfg-qualify-consts.after.mir

// START rustc.node18.SimplifyCfg-qualify-consts.after.mir
// fn main::{{closure}}(_1: [closure@NodeId(18) d:&'14mce D]) -> i32 {
//    let mut _0: i32;
//    let mut _2: i32;
//
//    bb0: {
//        StorageLive(_2);
//        _2 = ((*(_1.0: &'14mce D)).0: i32);
//        _0 = _2;
//        StorageDead(_2);
//        return;
//    }
// END rustc.node18.SimplifyCfg-qualify-consts.after.mir
