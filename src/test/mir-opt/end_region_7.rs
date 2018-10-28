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
// ignore-tidy-linelength

// Unwinding should EndRegion for in-scope borrows: Borrow of moved data.

fn main() {
    let d = D(0);
    foo(move || -> i32 { let r = &d; r.0 });
}

struct D(i32);
impl Drop for D { fn drop(&mut self) { println!("dropping D({})", self.0); } }

fn foo<F>(f: F) where F: FnOnce() -> i32 {
    if f() > 0 { panic!("im positive"); }
}

// END RUST SOURCE
// START rustc.main.SimplifyCfg-qualify-consts.after.mir
// fn main() -> () {
//     let mut _0: ();
//     ...
//     let _1: D;
//     ...
//     let mut _2: ();
//     let mut _3: [closure@NodeId(33) d:D];
//     bb0: {
//         StorageLive(_1);
//         _1 = D(const 0i32,);
//         FakeRead(ForLet, _1);
//         StorageLive(_3);
//         _3 = [closure@NodeId(33)] { d: move _1 };
//         _2 = const foo(move _3) -> [return: bb2, unwind: bb4];
//     }
//     bb1: {
//         resume;
//     }
//     bb2: {
//         drop(_3) -> [return: bb5, unwind: bb3];
//     }
//     bb3: {
//         drop(_1) -> bb1;
//     }
//     bb4: {
//         drop(_3) -> bb3;
//     }
//     bb5: {
//         StorageDead(_3);
//         _0 = ();
//         drop(_1) -> [return: bb6, unwind: bb1];
//     }
//     bb6: {
//         StorageDead(_1);
//         return;
//     }
// }
// END rustc.main.SimplifyCfg-qualify-consts.after.mir

// START rustc.main-{{closure}}.SimplifyCfg-qualify-consts.after.mir
// fn main::{{closure}}(_1: [closure@NodeId(33) d:D]) -> i32 {
//     let mut _0: i32;
//     ...
//     let _2: &'21_0rs D;
//     ...
//     bb0: {
//         StorageLive(_2);
//         _2 = &'21_0rs (_1.0: D);
//         FakeRead(ForLet, _2);
//         _0 = ((*_2).0: i32);
//         EndRegion('21_0rs);
//         StorageDead(_2);
//         drop(_1) -> [return: bb2, unwind: bb1];
//     }
//     bb1: {
//         resume;
//     }
//     bb2: {
//         return;
//     }
// }
// END rustc.main-{{closure}}.SimplifyCfg-qualify-consts.after.mir
