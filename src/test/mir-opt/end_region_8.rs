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

// Unwinding should EndRegion for in-scope borrows: Move of borrow into closure.

fn main() {
    let d = D(0);
    let r = &d;
    foo(move || -> i32 { r.0 });
}

struct D(i32);
impl Drop for D { fn drop(&mut self) { println!("dropping D({})", self.0); } }

fn foo<F>(f: F) where F: FnOnce() -> i32 {
    if f() > 0 { panic!("im positive"); }
}

// END RUST SOURCE
// START rustc.main.SimplifyCfg-qualify-consts.after.mir
// fn main() -> () {
//    let mut _0: ();
//    ...
//    let _2: &'26_1rs D;
//    ...
//    let _1: D;
//    ...
//    let mut _3: ();
//    let mut _4: [closure@NodeId(33) r:&'24s D];
//    bb0: {
//        StorageLive(_1);
//        _1 = D(const 0i32,);
//        FakeRead(ForLet, _1);
//        StorageLive(_2);
//        _2 = &'26_1rs _1;
//        FakeRead(ForLet, _2);
//        StorageLive(_4);
//        _4 = [closure@NodeId(33)] { r: _2 };
//        _3 = const foo(move _4) -> [return: bb2, unwind: bb3];
//    }
//    bb1: {
//        resume;
//    }
//    bb2: {
//        EndRegion('24s);
//        StorageDead(_4);
//        _0 = ();
//        EndRegion('26_1rs);
//        StorageDead(_2);
//        drop(_1) -> [return: bb4, unwind: bb1];
//    }
//    bb3: {
//        EndRegion('24s);
//        EndRegion('26_1rs);
//        drop(_1) -> bb1;
//    }
//    bb4: {
//        StorageDead(_1);
//        return;
//    }
// }
// END rustc.main.SimplifyCfg-qualify-consts.after.mir

// START rustc.main-{{closure}}.SimplifyCfg-qualify-consts.after.mir
// fn main::{{closure}}(_1: [closure@NodeId(33) r:&'24s D]) -> i32 {
//     let mut _0: i32;
//
//     bb0: {
//         _0 = ((*(_1.0: &'26_1rs D)).0: i32);
//         return;
//     }
// }
// END rustc.main-{{closure}}.SimplifyCfg-qualify-consts.after.mir
