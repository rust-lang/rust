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
fn nop<T>(_: T) {}

fn test_return_move_local() -> String {
    let x = String::new();
    x
}

fn test_return_copy_local() -> i32 {
    let x = 4;
    x
}

fn test_shadow_arg_mut(x: String) {
    let mut x = x;
    nop(&mut x);
}

fn test_return_field(x: &(i32, u32)) -> i32 {
    x.0
}

fn main() {
    // Make sure the functions actually get instantiated.
    test_return_move_local();
    test_return_copy_local();
    test_shadow_arg_mut(String::new());
    test_return_field(&(1, 2));
}

// END RUST SOURCE

// START rustc.test_return_move_local.SimplifyTempMoves.before.mir
//     _1 = const std::string::String::new() -> bb1;
//     ...
//     _2 = move _1;
//     _0 = move _2;
// END rustc.test_return_move_local.SimplifyTempMoves.before.mir
// START rustc.test_return_move_local.SimplifyTempMoves.after.mir
//     nop;
//     _0 = move _1;
// END rustc.test_return_move_local.SimplifyTempMoves.after.mir

// START rustc.test_return_copy_local.SimplifyTempMoves.before.mir
//     _1 = const 4i32;
//     ...
//     _2 = _1;
//     _0 = move _2;
// END rustc.test_return_copy_local.SimplifyTempMoves.before.mir
// START rustc.test_return_copy_local.SimplifyTempMoves.after.mir
//     nop;
//     _0 = _1;
// END rustc.test_return_copy_local.SimplifyTempMoves.after.mir

// START rustc.test_shadow_arg_mut.SimplifyTempMoves.before.mir
//     _3 = move _1;
//     _2 = move _3;
// END rustc.test_shadow_arg_mut.SimplifyTempMoves.before.mir
// START rustc.test_shadow_arg_mut.SimplifyTempMoves.after.mir
//     nop;
//     _2 = move _1;
// END rustc.test_shadow_arg_mut.SimplifyTempMoves.after.mir

// START rustc.test_return_field.SimplifyTempMoves.before.mir
//     _2 = ((*_1).0: i32);
//     _0 = move _2;
// END rustc.test_return_field.SimplifyTempMoves.before.mir
// START rustc.test_return_field.SimplifyTempMoves.after.mir
//     nop;
//     _0 = ((*_1).0: i32);
// END rustc.test_return_field.SimplifyTempMoves.after.mir
