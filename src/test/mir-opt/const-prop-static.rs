// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
static mut BANANA: u32 = 42;

fn assert_eq(x: u32, y: u32) {
    assert_eq!(x, y);
}

unsafe fn reset_banana() {
    BANANA = 42;
}

fn main() {
    unsafe {
        BANANA = 21;
        let x = BANANA + 21;
        if x != 42 { assert_eq(x, 42); }
        reset_banana();
        assert_eq(BANANA, 42);
    }
}
// END RUST SOURCE
// START rustc.node101.ConstPropagate.before.mir
// bb0: {
//     BANANA = const 21u32;
//     tmp0 = BANANA;
//     var0 = Add(tmp0, const 21u32);
//     tmp3 = var0;
//     tmp2 = Ne(tmp3, const 42u32);
//     if(tmp2) -> [true: bb1, false: bb2];
// }
//
// bb1: {
//     tmp5 = var0;
//     tmp4 = assert_eq(tmp5, const 42u32) -> bb3;
// }
//
// bb2: {
//     tmp1 = ();
//     goto -> bb4;
// }
//
// bb3: {
//     tmp1 = ();
//     goto -> bb4;
// }
//
// bb4: {
//     tmp6 = reset_banana() -> bb5;
// }
//
// bb5: {
//     tmp8 = BANANA;
//     tmp7 = assert_eq(tmp8, const 42u32) -> bb6;
// }
// END rustc.node101.ConstPropagate.before.mir
// START rustc.node101.DeadCode.after.mir
// bb0: {
//     BANANA = const 21u32;
//     goto -> bb1;
// }
//
// bb1: {
//     tmp6 = reset_banana() -> bb2;
// }
//
// bb2: {
//     tmp8 = BANANA;
//     tmp7 = assert_eq(tmp8, const 42u32) -> bb3;
// }
//
// bb3: {
//     return = ();
//     return;
// }
// END rustc.node101.DeadCode.after.mir
