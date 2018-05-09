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
//      _3 = discriminant(_2);
//      _14 = promoted[1];
//      _4 = &(*_14);
//      _9 = discriminant(_2);
//      switchInt(move _9) -> [0isize: bb5, 1isize: bb3, otherwise: bb7];
//  }
//  bb1: {
//      resume;
//  }
//  bb2: {  // arm1
//      _1 = (const 3i32, const 3i32);
//      goto -> bb13;
//  }
//  bb3: { // binding3(empty) and arm3
//      ReadForMatch(_4);
//      falseEdges -> [real: bb8, imaginary: bb4]; //pre_binding1
//  }
//  bb4: {
//      ReadForMatch(_4);
//      falseEdges -> [real: bb12, imaginary: bb5]; //pre_binding2
//  }
//  bb5: {
//      ReadForMatch(_4);
//      falseEdges -> [real: bb2, imaginary: bb6]; //pre_binding3
//  }
//  bb6: {
//      unreachable;
//  }
//  bb7: {
//      unreachable;
//  }
//  bb8: { // binding1 and guard
//      StorageLive(_7);
//      _13 = promoted[0];
//      _7 = &(((*_13) as Some).0: i32);
//      StorageLive(_10);
//      _10 = const guard() -> [return: bb9, unwind: bb1];
//  }
//  bb9: {
//      switchInt(move _10) -> [false: bb10, otherwise: bb11];
//  }
//  bb10: { // to pre_binding2
//      falseEdges -> [real: bb4, imaginary: bb4];
//  }
//  bb11: { // bindingNoLandingPads.before.mir2 and arm2
//      StorageLive(_5);
//      _5 = ((_2 as Some).0: i32);
//      StorageLive(_11);
//      _11 = _5;
//      _1 = (const 1i32, move _11);
//      StorageDead(_11);
//      goto -> bb13;
//  }
//  bb12: {
//      StorageLive(_8);
//      _8 = ((_2 as Some).0: i32);
//      StorageLive(_12);
//      _12 = _8;
//      _1 = (const 2i32, move_12);
//      StorageDead(_12);
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
//      _3 = discriminant(_2);
//      _4 = &_2;
//      _9 = discriminant(_2);
//      switchInt(move _9) -> [0isize: bb4, 1isize: bb3, otherwise: bb7];
//  }
//  bb1: {
//      resume;
//  }
//  bb2: { // arm2
//      _1 = (const 3i32, const 3i32);
//      goto -> bb13;
//  }
//  bb3: {
//      ReadForMatch(_4);
//      falseEdges -> [real: bb8, imaginary: bb4]; //pre_binding1
//  }
//  bb4: {
//      ReadForMatch(_4);
//      falseEdges -> [real: bb2, imaginary: bb5]; //pre_binding2
//  }
//  bb5: {
//      ReadForMatch(_4);
//      falseEdges -> [real: bb12, imaginary: bb6]; //pre_binding3
//  }
//  bb6: {
//      unreachable;
//  }
//  bb7: {
//      unreachable;
//  }
//  bb8: { // binding1 and guard
//      StorageLive(_7);
//      _7 = &((_2 as Some).0: i32);
//      StorageLive(_10);
//      _10 = const guard() -> [return: bb9, unwind: bb1];
//  }
//  bb9: { // end of guard
//      switchInt(move _10) -> [false: bb10, otherwise: bb11];
//  }
//  bb10: { // to pre_binding3 (can skip 2 since this is `Some`)
//      falseEdges -> [real: bb5, imaginary: bb4];
//  }
//  bb11: { // arm1
//      StorageLive(_5);
//      _5 = ((_2 as Some).0: i32);
//      StorageLive(_11);
//      _11 = _5;
//      _1 = (const 1i32, move _11);
//      StorageDead(_11);
//      goto -> bb13;
//  }
//  bb12: { // binding3 and arm3
//      StorageLive(_8);
//      _8 = ((_2 as Some).0: i32);
//      StorageLive(_12);
//      _12 = _8;
//      _1 = (const 2i32, move _12);
//      StorageDead(_12);
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
//     _3 = discriminant(_2);
//     _4 = &_2;
//     _13 = discriminant(_2);
//     switchInt(move _13) -> [1isize: bb2, otherwise: bb3];
// }
// bb1: {
//     resume;
// }
// bb2: {
//     ReadForMatch(_4);
//     falseEdges -> [real: bb7, imaginary: bb3]; //pre_binding1
// }
// bb3: {
//     ReadForMatch(_4);
//     falseEdges -> [real: bb11, imaginary: bb4]; //pre_binding2
// }
// bb4: {
//     ReadForMatch(_4);
//     falseEdges -> [real: bb12, imaginary: bb5]; //pre_binding3
// }
// bb5: {
//     ReadForMatch(_4);
//     falseEdges -> [real: bb16, imaginary: bb6]; //pre_binding4
// }
// bb6: {
//     unreachable;
// }
// bb7: { // binding1: Some(w) if guard()
//     StorageLive(_7);
//     _7 = &((_2 as Some).0: i32);
//     StorageLive(_14);
//     _14 = const guard() -> [return: bb8, unwind: bb1];
// }
// bb8: { //end of guard
//     switchInt(move _14) -> [false: bb9, otherwise: bb10];
// }
// bb9: { // to pre_binding2
//     falseEdges -> [real: bb3, imaginary: bb3];
// }
// bb10: { // set up bindings for arm1
//     StorageLive(_5);
//     _5 = ((_2 as Some).0: i32);
//     _1 = const 1i32;
//     goto -> bb17;
// }
// bb11: { // binding2 & arm2
//     StorageLive(_8);
//     _8 = _2;
//     _1 = const 2i32;
//     goto -> bb17;
// }
// bb12: { // binding3: Some(y) if guard2(y)
//     StorageLive(_11);
//     _11 = &((_2 as Some).0: i32);
//     StorageLive(_16);
//     StorageLive(_17);
//     _17 = (*_11);
//     _16 = const guard2(move _17) -> [return: bb13, unwind: bb1];
// }
// bb13: { // end of guard2
//     StorageDead(_17);
//     switchInt(move _16) -> [false: bb14, otherwise: bb15];
// }
// bb14: { // to pre_binding4
//     falseEdges -> [real: bb5, imaginary: bb5];
// }
// bb15: { // set up bindings for arm3
//     StorageLive(_9);
//     _9 = ((_2 as Some).0: i32);
//     _1 = const 3i32;
//     goto -> bb17;
// }
// bb16: { // binding4 & arm4
//     StorageLive(_12);
//     _12 = _2;
//     _1 = const 4i32;
//     goto -> bb17;
// }
// bb17: {
//     ...
//     return;
// }
// END rustc.main.QualifyAndPromoteConstants.before.mir
