// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[inline(never)]
fn mystery<T>(_: T) {}

fn return_arg(x: u8) -> u8 {
    x
}

fn assign_op(mut x: u8) -> u8 {
    x += 1;
    x
}

fn array_indexing(i: usize) {
    let a = [1, 2, 3];
    let _x = a[i]; // can remove the read, but need to keep the length assert
}

fn copy_to_locals(x: u8) -> u8 {
    let y = x; // cannot move
    let z = x; // last use; can move
    y + z
}

fn use_on_both_paths(b: bool, x: u8) -> u8 {
    // both uses are the last
    if b { x+1 } else { x-1 }
}

fn unneeded_rvalue(x: u8) -> u8 {
    let mut y = x;
    let z = y;
    y = 4; // no uses, so removable
    z
}

fn unneeded_complex_rvalue(x: u8) -> u8 {
    let mut y = x;
    let z = y;
    y = z+4+z; // no uses, so removable
    z
}

fn call_with_unneeded_result(x: u8) -> u8 {
    let y = mystery(4); // call might have side effects
    x
}

fn unneeded_drop_flag_update(b: bool) {
    let s = String::new();
    if b {
        mystery(s);
    }
}

fn loop_body(mut x: u8) {
    let mut y = 0;
    while y != 10 {
        x += 1;
        y = x; // not the last use because loop
    }
}

fn main() {
    // Make sure the functions actually get instantiated.
    return_arg(0);
    assign_op(0);
    array_indexing(0);
    copy_to_locals(0);
    use_on_both_paths(true, 1);
    unneeded_rvalue(0);
    unneeded_complex_rvalue(0);
    call_with_unneeded_result(0);
    unneeded_drop_flag_update(true);
    loop_body(0);
}

// ignore-tidy-linelength The assertions below are too long to fit, and can't wrap

// END RUST SOURCE

// START rustc.return_arg.WeakenLastUse.before.mir
//     _2 = _1;
//     _0 = move _2;
// END rustc.return_arg.WeakenLastUse.before.mir
// START rustc.return_arg.WeakenLastUse.after.mir
//     _2 = move _1;
//     _0 = move _2;
// END rustc.return_arg.WeakenLastUse.after.mir

// START rustc.assign_op.WeakenLastUse.before.mir
//     _1 = Add(_1, const 1u8);
// END rustc.assign_op.WeakenLastUse.before.mir
// START rustc.assign_op.WeakenLastUse.after.mir
//     _1 = Add(move _1, const 1u8);
// END rustc.assign_op.WeakenLastUse.after.mir

// START rustc.array_indexing.WeakenLastUse.before.mir
//     _5 = _1;
//     _6 = const 3usize;
//     _7 = Lt(_5, _6);
//     assert(move _7, "index out of bounds: the len is {} but the index is {}", move _6, _5) -> bb1;
// }
// bb1: {
//     _4 = _2[_5];
//     _3 = move _4;
// END rustc.array_indexing.WeakenLastUse.before.mir
// START rustc.array_indexing.WeakenLastUse.after.mir
//     _5 = move _1;
//     _6 = const 3usize;
//     _7 = Lt(_5, _6);
//     assert(move _7, "index out of bounds: the len is {} but the index is {}", move _6, move _5) -> bb1;
// }
// bb1: {
//     nop;
//     nop;
// END rustc.array_indexing.WeakenLastUse.after.mir

// START rustc.copy_to_locals.WeakenLastUse.before.mir
//     _3 = _1;
//     _2 = move _3;
//     ...
//     _5 = _1;
//     _4 = move _5;
//     ...
//     _6 = _2;
//     ...
//     _7 = _4;
//     ...
//     _0 = Add(move _6, move _7);
// END rustc.copy_to_locals.WeakenLastUse.before.mir
// START rustc.copy_to_locals.WeakenLastUse.after.mir
//     _3 = _1;
//     _2 = move _3;
//     ...
//     _5 = move _1;
//     _4 = move _5;
//     ...
//     _6 = move _2;
//     ...
//     _7 = move _4;
//     ...
//     _0 = Add(move _6, move _7);
// END rustc.copy_to_locals.WeakenLastUse.after.mir

// START rustc.use_on_both_paths.WeakenLastUse.before.mir
// bb0: {
//     StorageLive(_3);
//     _3 = _1;
//     switchInt(move _3) -> [0u8: bb2, otherwise: bb1];
// }
// bb1: {
//     StorageLive(_4);
//     _4 = _2;
//     _0 = Add(move _4, const 1u8);
//     StorageDead(_4);
//     goto -> bb3;
// }
// bb2: {
//     StorageLive(_5);
//     _5 = _2;
//     _0 = Sub(move _5, const 1u8);
//     StorageDead(_5);
//     goto -> bb3;
// }
// END rustc.use_on_both_paths.WeakenLastUse.before.mir
// START rustc.use_on_both_paths.WeakenLastUse.after.mir
//     _4 = move _2;
//     ...
//     _5 = move _2;
// END rustc.use_on_both_paths.WeakenLastUse.after.mir

// START rustc.unneeded_rvalue.WeakenLastUse.before.mir
//     _3 = _1;
//     _2 = move _3;
//     ...
//     _5 = _2;
//     _4 = move _5;
//     ...
//     _2 = const 4u8;
//     ...
//     _6 = _4;
//     _0 = move _6;
// END rustc.unneeded_rvalue.WeakenLastUse.before.mir
// START rustc.unneeded_rvalue.WeakenLastUse.after.mir
//     _3 = move _1;
//     _2 = move _3;
//     ...
//     _5 = move _2;
//     _4 = move _5;
//     ...
//     nop;
//     ...
//     _6 = move _4;
//     _0 = move _6;
// END rustc.unneeded_rvalue.WeakenLastUse.after.mir

// START rustc.unneeded_complex_rvalue.WeakenLastUse.before.mir
//     _3 = _1;
//     _2 = move _3;
//     ...
//     _5 = _2;
//     _4 = move _5;
//     ...
//     _7 = _4;
//     _6 = Add(move _7, const 4u8);
//     ...
//     _8 = _4;
//     _2 = Add(move _6, move _8);
//     ...
//     _9 = _4;
//     _0 = move _9;
// END rustc.unneeded_complex_rvalue.WeakenLastUse.before.mir
// START rustc.unneeded_complex_rvalue.WeakenLastUse.after.mir
//     _3 = move _1;
//     _2 = move _3;
//     ...
//     _5 = move _2;
//     _4 = move _5;
//     ...
//     nop;
//     nop;
//     ...
//     nop;
//     nop;
//     ...
//     _9 = move _4;
//     _0 = move _9;
// END rustc.unneeded_complex_rvalue.WeakenLastUse.after.mir

// START rustc.call_with_unneeded_result.WeakenLastUse.before.mir
// bb0: {
//     StorageLive(_2);
//     _2 = const mystery(const 4i32) -> bb1;
// }
// bb1: {
//     StorageLive(_3);
//     _3 = _1;
//     _0 = move _3;
//     StorageDead(_3);
//     StorageDead(_2);
//     return;
// }
// END rustc.call_with_unneeded_result.WeakenLastUse.before.mir
// START rustc.call_with_unneeded_result.WeakenLastUse.after.mir
//     _2 = const mystery(const 4i32) -> bb1;
// END rustc.call_with_unneeded_result.WeakenLastUse.after.mir

// START rustc.unneeded_drop_flag_update.WeakenLastUse.before.mir
// bb6: {
//     _6 = const false;
//     StorageDead(_2);
//     return;
// END rustc.unneeded_drop_flag_update.WeakenLastUse.before.mir
// START rustc.unneeded_drop_flag_update.WeakenLastUse.after.mir
// bb6: {
//     nop;
//     StorageDead(_2);
//     return;
// END rustc.unneeded_drop_flag_update.WeakenLastUse.after.mir

// START rustc.loop_body.WeakenLastUse.before.mir
//     _6 = _1;
//     _2 = move _6;
// END rustc.loop_body.WeakenLastUse.before.mir
// START rustc.loop_body.WeakenLastUse.after.mir
//     _6 = _1;
//     _2 = move _6;
// END rustc.loop_body.WeakenLastUse.after.mir
