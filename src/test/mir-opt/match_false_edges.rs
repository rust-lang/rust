// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z borrowck=mir

fn guard() -> bool {
    false
}

fn guard2(_:i32) -> bool {
    true
}

// no_mangle to make sure this gets instantiated even in an executable.
#[no_mangle]
pub fn full_tested_match() {
    let _ = match Some(42) {
        Some(x) if guard() => (1, x),
        Some(y) => (2, y),
        None => (3, 3),
    };
}

// no_mangle to make sure this gets instantiated even in an executable.
#[no_mangle]
pub fn full_tested_match2() {
    let _ = match Some(42) {
        Some(x) if guard() => (1, x),
        None => (3, 3),
        Some(y) => (2, y),
    };
}

fn main() {
    let _ = match Some(1) {
        Some(_w) if guard() => 1,
        _x => 2,
        Some(y) if guard2(y) => 3,
        _z => 4,
    };
}

// END RUST SOURCE
//
// START rustc.full_tested_match.QualifyAndPromoteConstants.after.mir
//  bb0: {
//      ...
//      _2 = std::option::Option<i32>::Some(const 42i32,);
//      _5 = discriminant(_2);
//      switchInt(move _5) -> [0isize: bb6, 1isize: bb4, otherwise: bb8];
//  }
//  bb1: {
//      resume;
//  }
//  bb2: {  // arm1
//      StorageLive(_7);
//      _7 = _3;
//      _1 = (const 1i32, move _7);
//      StorageDead(_7);
//      goto -> bb13;
//  }
//  bb3: { // binding3(empty) and arm3
//      _1 = (const 3i32, const 3i32);
//      goto -> bb13;
//  }
//  bb4: {
//      falseEdges -> [real: bb9, imaginary: bb5]; //pre_binding1
//  }
//  bb5: {
//      falseEdges -> [real: bb12, imaginary: bb6]; //pre_binding2
//  }
//  bb6: {
//      falseEdges -> [real: bb3, imaginary: bb7]; //pre_binding3
//  }
//  bb7: {
//      unreachable;
//  }
//  bb8: {
//      unreachable;
//  }
//  bb9: { // binding1 and guard
//      StorageLive(_3);
//      _3 = ((_2 as Some).0: i32);
//      StorageLive(_6);
//      _6 = const guard() -> [return: bb10, unwind: bb1];
//  }
//  bb10: { // end of guard
//      switchInt(move _6) -> [0u8: bb11, otherwise: bb2];
//  }
//  bb11: { // to pre_binding2
//      falseEdges -> [real: bb5, imaginary: bb5];
//  }
//  bb12: { // bindingNoLandingPads.before.mir2 and arm2
//      StorageLive(_4);
//      _4 = ((_2 as Some).0: i32);
//      StorageLive(_8);
//      _8 = _4;
//      _1 = (const 2i32, move _8);
//      StorageDead(_8);
//      goto -> bb13;
//  }
//  bb13: {
//      ...
//      return;
//  }
// END rustc.full_tested_match.QualifyAndPromoteConstants.after.mir
//
// START rustc.full_tested_match2.QualifyAndPromoteConstants.before.mir
//  bb0: {
//      ...
//      _2 = std::option::Option<i32>::Some(const 42i32,);
//      _5 = discriminant(_2);
//      switchInt(move _5) -> [0isize: bb5, 1isize: bb4, otherwise: bb8];
//  }
//  bb1: {
//      resume;
//  }
//  bb2: { // arm1
//      StorageLive(_7);
//      _7 = _3;
//      _1 = (const 1i32, move _7);
//      StorageDead(_7);
//      goto -> bb13;
//  }
//  bb3: { // binding3(empty) and arm3
//      _1 = (const 3i32, const 3i32);
//      goto -> bb13;
//  }
//  bb4: {
//      falseEdges -> [real: bb9, imaginary: bb5]; //pre_binding1
//  }
//  bb5: {
//      falseEdges -> [real: bb3, imaginary: bb6]; //pre_binding2
//  }
//  bb6: {
//      falseEdges -> [real: bb12, imaginary: bb7]; //pre_binding3
//  }
//  bb7: {
//      unreachable;
//  }
//  bb8: {
//      unreachable;
//  }
//  bb9: { // binding1 and guard
//      StorageLive(_3);
//      _3 = ((_2 as Some).0: i32);
//      StorageLive(_6);
//      _6 = const guard() -> [return: bb10, unwind: bb1];
//  }
//  bb10: { // end of guard
//      switchInt(move _6) -> [0u8: bb11, otherwise: bb2];
//  }
//  bb11: { // to pre_binding2
//      falseEdges -> [real: bb6, imaginary: bb5];
//  }
//  bb12: { // binding2 and arm2
//      StorageLive(_4);
//      _4 = ((_2 as Some).0: i32);
//      StorageLive(_8);
//      _8 = _4;
//      _1 = (const 2i32, move _8);
//      StorageDead(_8);
//      goto -> bb13;
//  }
//  bb13: {
//      ...
//      return;
//  }
// END rustc.full_tested_match2.QualifyAndPromoteConstants.before.mir
//
// START rustc.main.QualifyAndPromoteConstants.before.mir
// bb0: {
//     ...
//     _2 = std::option::Option<i32>::Some(const 1i32,);
//     _7 = discriminant(_2);
//     switchInt(move _7) -> [1isize: bb4, otherwise: bb5];
// }
// bb1: {
//     resume;
// }
// bb2: { // arm1
//     _1 = const 1i32;
//     goto -> bb17;
// }
// bb3: { // arm3
//     _1 = const 3i32;
//     goto -> bb17;
// }
//
// bb4: {
//     falseEdges -> [real: bb9, imaginary: bb5]; //pre_binding1
// }
// bb5: {
//     falseEdges -> [real: bb12, imaginary: bb6]; //pre_binding2
// }
// bb6: {
//     falseEdges -> [real: bb13, imaginary: bb7]; //pre_binding3
// }
// bb7: {
//     falseEdges -> [real: bb16, imaginary: bb8]; //pre_binding4
// }
// bb8: {
//     unreachable;
// }
// bb9: { // binding1: Some(w) if guard()
//     StorageLive(_3);
//     _3 = ((_2 as Some).0: i32);
//     StorageLive(_8);
//     _8 = const guard() -> [return: bb10, unwind: bb1];
// }
// bb10: { //end of guard
//    switchInt(move _8) -> [0u8: bb11, otherwise: bb2];
// }
// bb11: { // to pre_binding2
//     falseEdges -> [real: bb5, imaginary: bb5];
// }
// bb12: { // binding2 & arm2
//     StorageLive(_4);
//     _4 = _2;
//     _1 = const 2i32;
//     goto -> bb17;
// }
// bb13: { // binding3: Some(y) if guard2(y)
//     StorageLive(_5);
//     _5 = ((_2 as Some).0: i32);
//     StorageLive(_10);
//     StorageLive(_11);
//     _11 = _5;
//     _10 = const guard2(move _11) -> [return: bb14, unwind: bb1];
// }
// bb14: { // end of guard2
//     StorageDead(_11);
//     switchInt(move _10) -> [0u8: bb15, otherwise: bb3];
// }
// bb15: { // to pre_binding4
//     falseEdges -> [real: bb7, imaginary: bb7];
// }
// bb16: { // binding4 & arm4
//     StorageLive(_6);
//     _6 = _2;
//     _1 = const 4i32;
//     goto -> bb17;
// }
// bb17: {
//     ...
//     return;
// }
// END rustc.main.QualifyAndPromoteConstants.before.mir
