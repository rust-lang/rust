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

// This test models a scenario with a cyclic reference. Rust obviously
// needs to handle such cases.
//
// The interesting part about this test is that such case shows that
// one cannot generally force all references to be dead before you hit
// their EndRegion; at least, not without breaking the more important
// property that all borrowed storage locations have their regions
// ended strictly before their StorageDeads. (This test was inspired
// by discussion on Issue #43481.)

use std::cell::Cell;

struct S<'a> {
    r: Cell<Option<&'a S<'a>>>,
}

fn main() {
    loop {
        let x = S { r: Cell::new(None) };
        x.r.set(Some(&x));
        if query() { break; }
        x.r.set(Some(&x));
    }
}

fn query() -> bool { true }

// END RUST SOURCE
// START rustc.main.SimplifyCfg-qualify-consts.after.mir
// fn main() -> (){
//     let mut _0: ();
//     scope 1 {
//     }
//     scope 2 {
//         let _2: S<'36_0rs>;
//     }
//     let mut _1: ();
//     let mut _3: std::cell::Cell<std::option::Option<&'36_0rs S<'36_0rs>>>;
//     let mut _4: std::option::Option<&'36_0rs S<'36_0rs>>;
//     let mut _5: ();
//     let mut _6: &'17s std::cell::Cell<std::option::Option<&'36_0rs S<'36_0rs>>>;
//     let mut _7: std::option::Option<&'36_0rs S<'36_0rs>>;
//     let mut _8: &'36_0rs S<'36_0rs>;
//     let mut _9: &'36_0rs S<'36_0rs>;
//     let mut _10: ();
//     let mut _11: bool;
//     let mut _12: !;
//     let mut _13: ();
//     let mut _14: &'34s std::cell::Cell<std::option::Option<&'36_0rs S<'36_0rs>>>;
//     let mut _15: std::option::Option<&'36_0rs S<'36_0rs>>;
//     let mut _16: &'36_0rs S<'36_0rs>;
//     let mut _17: &'36_0rs S<'36_0rs>;
//     bb0: {
//         goto -> bb1;
//     }
//     bb1: {
//         falseUnwind -> [real: bb2, cleanup: bb3];
//     }
//     bb2: {
//         StorageLive(_2);
//         StorageLive(_3);
//         StorageLive(_4);
//         _4 = std::option::Option<&'36_0rs S<'36_0rs>>::None;
//         _3 = const <std::cell::Cell<T>>::new(move _4) -> [return: bb4, unwind: bb3];
//     }
//     bb3: {
//         resume;
//     }
//     bb4: {
//         StorageDead(_4);
//         _2 = S<'36_0rs> { r: move _3 };
//         StorageDead(_3);
//         StorageLive(_6);
//         _6 = &'17s (_2.0: std::cell::Cell<std::option::Option<&'36_0rs S<'36_0rs>>>);
//         StorageLive(_7);
//         StorageLive(_8);
//         StorageLive(_9);
//         _9 = &'36_0rs _2;
//         _8 = &'36_0rs (*_9);
//         _7 = std::option::Option<&'36_0rs S<'36_0rs>>::Some(move _8,);
//         StorageDead(_8);
//         _5 = const <std::cell::Cell<T>>::set(move _6, move _7) -> [return: bb5, unwind: bb3];
//     }
//     bb5: {
//         EndRegion('17s);
//         StorageDead(_7);
//         StorageDead(_6);
//         StorageDead(_9);
//         StorageLive(_11);
//         _11 = const query() -> [return: bb6, unwind: bb3];
//     }
//     bb6: {
//         switchInt(move _11) -> [false: bb8, otherwise: bb7];
//     }
//     bb7: {
//         _0 = ();
//         StorageDead(_11);
//         EndRegion('36_0rs);
//         StorageDead(_2);
//         return;
//     }
//     bb8: {
//         _10 = ();
//         StorageDead(_11);
//         StorageLive(_14);
//         _14 = &'34s (_2.0: std::cell::Cell<std::option::Option<&'36_0rs S<'36_0rs>>>);
//         StorageLive(_15);
//         StorageLive(_16);
//         StorageLive(_17);
//         _17 = &'36_0rs _2;
//         _16 = &'36_0rs (*_17);
//         _15 = std::option::Option<&'36_0rs S<'36_0rs>>::Some(move _16,);
//         StorageDead(_16);
//         _13 = const <std::cell::Cell<T>>::set(move _14, move _15) -> [return: bb9, unwind: bb3];
//     }
//     bb9: {
//         EndRegion('34s);
//         StorageDead(_15);
//         StorageDead(_14);
//         StorageDead(_17);
//         _1 = ();
//         EndRegion('36_0rs);
//         StorageDead(_2);
//         goto -> bb1;
//     }
// }
// END rustc.main.SimplifyCfg-qualify-consts.after.mir
