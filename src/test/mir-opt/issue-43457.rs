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

// Regression test for #43457: an `EndRegion` was missing from output
// because compiler was using a faulty means for region map lookup.

use std::cell::RefCell;

fn rc_refcell_test(r: RefCell<i32>) {
    r.borrow_mut();
}

fn main() { }

// END RUST SOURCE
// START rustc.rc_refcell_test.SimplifyCfg-qualify-consts.after.mir
//
// fn rc_refcell_test(_1: std::cell::RefCell<i32>) -> () {
//     let mut _0: ();
//     scope 1 {
//         let _2: std::cell::RefCell<i32>;
//     }
//     let mut _3: std::cell::RefMut<'17ds, i32>;
//     let mut _4: &'17ds std::cell::RefCell<i32>;
//
//     bb0: {
//         StorageLive(_2);
//         _2 = _1;
//         StorageLive(_4);
//         _4 = &'17ds _2;
//         _3 = const <std::cell::RefCell<T>>::borrow_mut(_4) -> bb1;
//     }
//
//     bb1: {
//         drop(_3) -> bb2;
//     }
//
//     bb2: {
//         StorageDead(_4);
//         EndRegion('17ds);
//         _0 = ();
//         StorageDead(_2);
//         return;
//     }
// }
