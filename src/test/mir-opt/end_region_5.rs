// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z identify_regions -Z span_free_formats -Z emit-end-regions

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
// START rustc.main.SimplifyCfg-qualify-consts.after.mir
// fn main() -> () {
//     ...
//     let mut _0: ();
//     ...
//     let _1: D;
//     ...
//     let mut _2: ();
//     let mut _3: [closure@NodeId(18) d:&'14s D];
//     let mut _4: &'14s D;
//     bb0: {
//         StorageLive(_1);
//         _1 = D::{{constructor}}(const 0i32,);
//         StorageLive(_3);
//         StorageLive(_4);
//         _4 = &'14s _1;
//         _3 = [closure@NodeId(18)] { d: move _4 };
//         StorageDead(_4);
//         _2 = const foo(move _3) -> [return: bb2, unwind: bb3];
//     }
//     bb1: {
//         resume;
//     }
//     bb2: {
//         EndRegion('14s);
//         StorageDead(_3);
//         _0 = ();
//         drop(_1) -> [return: bb4, unwind: bb1];
//     }
//     bb3: {
//         EndRegion('14s);
//         drop(_1) -> bb1;
//     }
//     bb4: {
//         StorageDead(_1);
//         return;
//     }
// }
// END rustc.main.SimplifyCfg-qualify-consts.after.mir

// START rustc.main-{{closure}}.SimplifyCfg-qualify-consts.after.mir
// fn main::{{closure}}(_1: [closure@NodeId(18) d:&'14s D]) -> i32 {
//    let mut _0: i32;
//    let mut _2: i32;
//
//    bb0: {
//        StorageLive(_2);
//        _2 = ((*(_1.0: &'14s D)).0: i32);
//        _0 = move _2;
//        StorageDead(_2);
//        return;
//    }
// END rustc.main-{{closure}}.SimplifyCfg-qualify-consts.after.mir
