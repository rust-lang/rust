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
        Some(_) if guard() => 1,
        Some(_) => 2,
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
//      _3 = discriminant(_2);
//      switchInt(_3) -> [0isize: bb8, otherwise: bb9];
//  }
//  bb1: { // arm1
//      _1 = const 1i32;
//      goto -> bb13;
//  }
//  bb2: { // arm2
//      _1 = const 2i32;
//      goto -> bb13;
//  }
//  bb3: { // arm3
//      _1 = const 3i32;
//      goto -> bb13;
//  }
//  bb4: { // binding1
//      ...
//      _4 = const guard() -> bb10;
//  }
//  bb5: { // binding2
//      falseEdges -> [real: bb2, imaginary: bb6];
//  }
//  bb6: { // binding3
//      falseEdges -> [real: bb3, imaginary: bb7];
//  }
//  bb7: {
//      unreachable;
//  }
//  bb8: {
//      falseEdges -> [real: bb6, imaginary: bb7]; // from before_binding3 to unreachable
//  }
//  bb9: {
//      falseEdges -> [real: bb4, imaginary: bb5]; // from before_binding1 to binding2
//  }
//  bb10: {
//      switchInt(_4) -> [0u8: bb11, otherwise: bb1]; // end of guard
//  }
//  bb11: {
//      falseEdges -> [real: bb12, imaginary: bb5]; // after_guard to binding2
//  }
//  bb12: {
//      falseEdges -> [real: bb5, imaginary: bb6]; // from before_binding2 to binding3
//  }
//  bb13: {
//      ...
//      return;
//  }
//
//
// END rustc.node17.NLL.before.mir
//
// START rustc.node36.NLL.before.mir
// bb0: {
//     ...
//     _2 = std::option::Option<i32>::Some(const 1i32,);
//     _7 = discriminant(_2);
//     switchInt(_7) -> [1isize: bb10, otherwise: bb13];
// }
// bb1: { // arm1
//      _1 = const 1i32;
//      goto -> bb17;
// }
// bb2: { // arm2
//     _1 = const 2i32;
//     goto -> bb17;
// }
// bb3: { // arm3
//     _1 = const 3i32;
//      goto -> bb17;
// }
// bb4: { // arm4
//     _1 = const 4i32;
//      goto -> bb17;
// }
// bb5: { // binding1: Some(w) if guard() =>
//     ...
//     _8 = const guard() -> bb11;
// }
// bb6: { // binding2: x =>
//     ...
//     _4 = _2;
//     falseEdges -> [real: bb2, imaginary: bb7]; // after binding2 to binding3
//    }
// bb7: { // binding3: Some(y) if guard2(y) =>
//     ...
//     _10 = const guard2(_11) -> bb14;
// }
// bb8: { // binding4: z_ =>
//     ...
//     _6 = _2;
//     falseEdges -> [real: bb4, imaginary: bb9]; // after binding3 to unreachable
// }
// bb9: {
//     unreachable;
// }
// bb10: {
//     falseEdges -> [real: bb5, imaginary: bb6]; // from before_binding1 to binding2
// }
// bb11: {
//     switchInt(_8) -> [0u8: bb12, otherwise: bb1]; // end of gurard
// }
// bb12: {
//     falseEdges -> [real: bb13, imaginary: bb6]; // after guard to binding2
// }
// bb13: {
//     falseEdges -> [real: bb6, imaginary: bb7]; // from before_binding2 to binding3
// }
// bb14: {
//      ...
//      switchInt(_10) -> [0u8: bb15, otherwise: bb3]; // end of guard2
// }
// bb15: {
//     falseEdges -> [real: bb16, imaginary: bb8]; // after guard2 to binding4
// }
// bb16: {
//     falseEdges -> [real: bb8, imaginary: bb9]; // from befor binding3 to binding4
// }
// bb17: {
//     ...
//     return;
// }
// END rustc.node36.NLL.before.mir
