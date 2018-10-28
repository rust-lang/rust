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

// A scenario with significant destruction code extents (which have
// suffix "dce" in current `-Z identify_regions` rendering).

#![feature(dropck_eyepatch)]

fn main() {
    // Since the second param to `D1` is may_dangle, it is legal for
    // the region of that parameter to end before the drop code for D1
    // is executed.
    (D1(&S1("ex1"), &S1("dang1"))).0;
}

#[derive(Debug)]
struct S1(&'static str);

#[derive(Debug)]
struct D1<'a, 'b>(&'a S1, &'b S1);

// The `#[may_dangle]` means that references of type `&'b _` may be
// invalid during the execution of this destructor; i.e. in this case
// the destructor code is not allowed to read or write `*self.1`, while
// it can read/write `*self.0`.
unsafe impl<'a, #[may_dangle] 'b> Drop for D1<'a, 'b> {
    fn drop(&mut self) {
        println!("D1({:?}, _)", self.0);
    }
}

// Notes on the MIR output below:
//
// 1. The `EndRegion('13s)` is allowed to precede the `drop(_3)`
//    solely because of the #[may_dangle] mentioned above.
//
// 2. Regarding the occurrence of `EndRegion('15ds)` *after* `StorageDead(_6)`
//    (where we have borrows `&'15ds _6`): Eventually:
//
//    i. this code should be rejected (by mir-borrowck), or
//
//    ii. the MIR code generation should be changed so that the
//        EndRegion('15ds)` precedes `StorageDead(_6)` in the
//        control-flow.  (Note: arielb1 views drop+storagedead as one
//        unit, and does not see this option as a useful avenue to
//        explore.), or
//
//    iii. the presence of EndRegion should be made irrelevant by a
//        transformation encoding the effects of rvalue-promotion.
//        This may be the simplest and most-likely option; note in
//        particular that `StorageDead(_6)` goes away below in
//        rustc.main.QualifyAndPromoteConstants.after.mir

// END RUST SOURCE

// START rustc.main.QualifyAndPromoteConstants.before.mir
// fn main() -> () {
// let mut _0: ();
//     let mut _1: &'15ds S1;
//     let mut _2: D1<'15ds, '13s>;
//     let mut _3: &'15ds S1;
//     let mut _4: &'15ds S1;
//     let _5: S1;
//     let mut _6: &'13s S1;
//     let mut _7: &'13s S1;
//     let _8: S1;
//     bb0: {
//         StorageLive(_2);
//         StorageLive(_3);
//         StorageLive(_4);
//         StorageLive(_5);
//         _5 = S1(const "ex1",);
//         _4 = &'15ds _5;
//         _3 = &'15ds (*_4);
//         StorageLive(_6);
//         StorageLive(_7);
//         StorageLive(_8);
//         _8 = S1(const "dang1",);
//         _7 = &'13s _8;
//         _6 = &'13s (*_7);
//         _2 = D1<'15ds, '13s>(move _3, move _6);
//         EndRegion('13s);
//         StorageDead(_6);
//         StorageDead(_3);
//         _1 = (_2.0: &'15ds S1);
//         drop(_2) -> [return: bb2, unwind: bb1];
//     }
//     bb1: {
//         resume;
//     }
//     bb2: {
//         StorageDead(_2);
//         StorageDead(_7);
//         StorageDead(_8);
//         StorageDead(_4);
//         StorageDead(_5);
//         EndRegion('15ds);
//         _0 = ();
//         return;
//     }
// }
// END rustc.main.QualifyAndPromoteConstants.before.mir

// START rustc.main.QualifyAndPromoteConstants.after.mir
// fn main() -> (){
//     let mut _0: ();
//     let mut _1: &'15ds S1;
//     let mut _2: D1<'15ds, '13s>;
//     let mut _3: &'15ds S1;
//     let mut _4: &'15ds S1;
//     let _5: S1;
//     let mut _6: &'13s S1;
//     let mut _7: &'13s S1;
//     let _8: S1;
//     bb0: {
//         StorageLive(_2);
//         StorageLive(_3);
//         StorageLive(_4);
//         _4 = &'15ds (promoted[1]: S1);
//         _3 = &'15ds (*_4);
//         StorageLive(_6);
//         StorageLive(_7);
//         _7 = &'13s (promoted[0]: S1);
//         _6 = &'13s (*_7);
//         _2 = D1<'15ds, '13s>(move _3, move _6);
//         EndRegion('13s);
//         StorageDead(_6);
//         StorageDead(_3);
//         _1 = (_2.0: &'15ds S1);
//         drop(_2) -> [return: bb2, unwind: bb1];
//     }
//     bb1: {
//         resume;
//     }
//     bb2: {
//         StorageDead(_2);
//         StorageDead(_7);
//         StorageDead(_4);
//         EndRegion('15ds);
//         _0 = ();
//         return;
//     }
// }
// END rustc.main.QualifyAndPromoteConstants.after.mir
