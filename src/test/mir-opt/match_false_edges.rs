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

fn full_tested_match()
{
    let _ = match Some(42) {
        Some(x) if guard() => 1 + x,
        Some(y) => 2 + y,
        None => 3
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
// START rustc.node17.NLL.before.mir
//  bb0: {
//      ...
//      _2 = std::option::Option<i32>::Some(const 42i32,);
//      _5 = discriminant(_2);
//      switchInt(_5) -> [0isize: bb6, otherwise: bb7];
//  }
//  bb1: { // arm1
//      StorageLive(_7);
//      _7 = _3;
//      _1 = Add(const 1i32, _7);
//      ...
//      goto -> bb11;
//  }
//  bb2: { // binding1 guard
//      StorageLive(_3);
//      _3 = ((_2 as Some).0: i32);
//      StorageLive(_6);
//      _6 = const guard() -> bb8;
//  }
//  bb3: { // binding2 & arm2
//      StorageLive(_4);
//      _4 = ((_2 as Some).0: i32);
//      StorageLive(_8);
//      _8 = _4;
//      _1 = Add(const 2i32, _8);
//      StorageDead(_8);
//      goto -> bb11;
//  }
//  bb4: { // binding3(empty) arm3
//      _1 = const 3i32;
//      goto -> bb11;
//  }
//  bb5: {
//      unreachable;
//  }
//  bb6: {
//      falseEdges -> [real: bb4, imaginary: bb5]; // from before_binding3 to unreachable
//  }
//  bb7: {
//      falseEdges -> [real: bb2, imaginary: bb3]; // from before_binding1 to binding2
//  }
//  bb8: {
//      switchInt(_6) -> [0u8: bb9, otherwise: bb1]; // end of guard
//  }
//  bb9: {
//      falseEdges -> [real: bb10, imaginary: bb3]; // after_guard to binding2
//  }
//  bb10: {
//      falseEdges -> [real: bb3, imaginary: bb4]; // from before_binding2 to binding3
//  }
//  bb11: {
//      ...
//      return;
//  }
//
//
// END rustc.node17.NLL.before.mir
//
// START rustc.node40.NLL.before.mir
// bb0: {
//     ...
//     _2 = std::option::Option<i32>::Some(const 1i32,);
//     _7 = discriminant(_2);
//     switchInt(_7) -> [1isize: bb8, otherwise: bb11];
// }
// bb1: { // arm1
//      _1 = const 1i32;
//      goto -> bb15;
// }
// bb2: { // arm3
//     _1 = const 3i32;
//      goto -> bb15;
// }
// bb3: { // binding1: Some(w) if guard() =>
//     StorageLive(_3);
//     _3 = ((_2 as Some).0: i32);
//     StorageLive(_8);
//     _8 = const guard() -> bb9;
// }
// bb4: { // binding2 & arm2
//     StorageLive(_4);
//     _4 = _2;
//     _1 = const 2i32;
//     goto -> bb15;
// }
// bb5: { // binding3: Some(y) if guard2(y) =>
//     StorageLive(_5);
//     _5 = ((_2 as Some).0: i32);
//     StorageLive(_10);
//     StorageLive(_11);
//     _11 = _5;
//    _10 = const guard2(_11) -> bb12;
// }
// bb6: { // binding4 & arm4
//     StorageLive(_6);
//     _6 = _2;
//     _1 = const 4i32;
//     goto -> bb15;
// }
// bb7: {
//     unreachable;
// }
// bb8: {
//     falseEdges -> [real: bb3, imaginary: bb4]; // from before_binding1 to binding2
// }
// bb9: {
//     switchInt(_8) -> [0u8: bb10, otherwise: bb1]; // end of gurard
// }
// bb10: {
//     falseEdges -> [real: bb11, imaginary: bb4]; // after guard to binding2
// }
// bb11: {
//     falseEdges -> [real: bb4, imaginary: bb5]; // from before_binding2 to binding3
// }
// bb12: {
//      StorageDead(_11);
//      switchInt(_10) -> [0u8: bb13, otherwise: bb2]; // end of guard2
// }
// bb13: {
//     falseEdges -> [real: bb14, imaginary: bb6]; // after guard2 to binding4
// }
// bb14: {
//     falseEdges -> [real: bb6, imaginary: bb7]; // from befor binding4 to unreachable
// }
// bb15: {
//     ...
//     return;
// }
// END rustc.node40.NLL.before.mir
