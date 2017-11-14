// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z emit-end-regions -Z borrowck-mir

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
// START rustc.full_tested_match.SimplifyBranches-initial.before.mir
//  bb0: {
//      ...
//      _2 = std::option::Option<i32>::Some(const 42i32,);
//      _5 = discriminant(_2);
//      switchInt(_5) -> [0isize: bb5, 1isize: bb3, otherwise: bb7];
//  }
//  bb1: { // arm1
//      StorageLive(_7);
//      _7 = _3;
//      _1 = (const 1i32, _7);
//      StorageDead(_7);
//      goto -> bb12;
//  }
//  bb2: { // binding3(empty) and arm3
//      _1 = (const 3i32, const 3i32);
//      goto -> bb12;
//  }
//  bb3: {
//      falseEdges -> [real: bb8, imaginary: bb4]; //pre_binding1
//  }
//  bb4: {
//      falseEdges -> [real: bb11, imaginary: bb5]; //pre_binding2
//  }
//  bb5: {
//      falseEdges -> [real: bb2, imaginary: bb6]; //pre_binding3
//  }
//  bb6: {
//      unreachable;
//  }
//  bb7: {
//      unreachable;
//  }
//  bb8: { // binding1 and guard
//      StorageLive(_3);
//      _3 = ((_2 as Some).0: i32);
//      StorageLive(_6);
//      _6 = const guard() -> bb9;
//  }
//  bb9: { // end of guard
//      switchInt(_6) -> [0u8: bb10, otherwise: bb1];
//  }
//  bb10: { // to pre_binding2
//      falseEdges -> [real: bb4, imaginary: bb4];
//  }
//  bb11: { // bindingNoLandingPads.before.mir2 and arm2
//      StorageLive(_4);
//      _4 = ((_2 as Some).0: i32);
//      StorageLive(_8);
//      _8 = _4;
//      _1 = (const 2i32, _8);
//      StorageDead(_8);
//      goto -> bb12;
//  }
//  bb12: {
//      ...
//      return;
//  }
// END rustc.full_tested_match.SimplifyBranches-initial.before.mir
//
// START rustc.full_tested_match2.SimplifyBranches-initial.before.mir
//  bb0: {
//      ...
//      _2 = std::option::Option<i32>::Some(const 42i32,);
//      _5 = discriminant(_2);
//      switchInt(_5) -> [0isize: bb4, 1isize: bb3, otherwise: bb7];
//  }
//  bb1: { // arm1
//      StorageLive(_7);
//      _7 = _3;
//      _1 = (const 1i32, _7);
//      StorageDead(_7);
//      goto -> bb12;
//  }
//  bb2: { // binding3(empty) and arm3
//      _1 = (const 3i32, const 3i32);
//      goto -> bb12;
//  }
//  bb3: {
//      falseEdges -> [real: bb8, imaginary: bb4]; //pre_binding1
//  }
//  bb4: {
//      falseEdges -> [real: bb2, imaginary: bb5]; //pre_binding2
//  }
//  bb5: {
//      falseEdges -> [real: bb11, imaginary: bb6]; //pre_binding3
//  }
//  bb6: {
//      unreachable;
//  }
//  bb7: {
//      unreachable;
//  }
//  bb8: { // binding1 and guard
//      StorageLive(_3);
//      _3 = ((_2 as Some).0: i32);
//      StorageLive(_6);
//      _6 = const guard() -> bb9;
//  }
//  bb9: { // end of guard
//      switchInt(_6) -> [0u8: bb10, otherwise: bb1];
//  }
//  bb10: { // to pre_binding2
//      falseEdges -> [real: bb5, imaginary: bb4];
//  }
//  bb11: { // binding2 and arm2
//      StorageLive(_4);
//      _4 = ((_2 as Some).0: i32);
//      StorageLive(_8);
//      _8 = _4;
//      _1 = (const 2i32, _8);
//      StorageDead(_8);
//      goto -> bb12;
//  }
//  bb12: {
//      ...
//      return;
//  }
// END rustc.full_tested_match2.SimplifyBranches-initial.before.mir
//
// START rustc.main.SimplifyBranches-initial.before.mir
// bb0: {
//     ...
//     _2 = std::option::Option<i32>::Some(const 1i32,);
//     _7 = discriminant(_2);
//     switchInt(_7) -> [1isize: bb3, otherwise: bb4];
// }
// bb1: { // arm1
//      _1 = const 1i32;
//      goto -> bb16;
// }
// bb2: { // arm3
//     _1 = const 3i32;
//      goto -> bb16;
// }
//
//  bb3: {
//      falseEdges -> [real: bb8, imaginary: bb4]; //pre_binding1
//  }
//  bb4: {
//      falseEdges -> [real: bb11, imaginary: bb5]; //pre_binding2
//  }
//  bb5: {
//      falseEdges -> [real: bb12, imaginary: bb6]; //pre_binding3
//  }
//  bb6: {
//      falseEdges -> [real: bb15, imaginary: bb7]; //pre_binding4
//  }
//  bb7: {
//      unreachable;
//  }
//  bb8: { // binding1: Some(w) if guard()
//      StorageLive(_3);
//      _3 = ((_2 as Some).0: i32);
//      StorageLive(_8);
//      _8 = const guard() -> bb9;
//  }
//  bb9: { //end of guard
//      switchInt(_8) -> [0u8: bb10, otherwise: bb1];
//  }
//  bb10: { // to pre_binding2
//      falseEdges -> [real: bb4, imaginary: bb4];
//  }
//  bb11: { // binding2 & arm2
//      StorageLive(_4);
//      _4 = _2;
//      _1 = const 2i32;
//      goto -> bb16;
//  }
//  bb12: { // binding3: Some(y) if guard2(y)
//      StorageLive(_5);
//      _5 = ((_2 as Some).0: i32);
//      StorageLive(_10);
//      StorageLive(_11);
//      _11 = _5;
//      _10 = const guard2(_11) -> bb13;
//  }
//  bb13: { // end of guard2
//      StorageDead(_11);
//      switchInt(_10) -> [0u8: bb14, otherwise: bb2];
//  }
//  bb14: { // to pre_binding4
//      falseEdges -> [real: bb6, imaginary: bb6];
//  }
//  bb15: { // binding4 & arm4
//      StorageLive(_6);
//      _6 = _2;
//      _1 = const 4i32;
//      goto -> bb16;
//  }
// bb16: {
//     ...
//     return;
// }
// END rustc.main.SimplifyBranches-initial.before.mir
