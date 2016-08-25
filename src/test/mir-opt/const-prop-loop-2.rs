// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main(){
    let mut x = 0;
    let mut y = true;
    while y {
        if x != 0 {
            x = 1;
        }
        y = false;
    }
    println(x);
}

fn println(x: u32) {
    println!("{}", x);
}
// END RUST SOURCE
// START rustc.node4.ConstPropagate.before.mir
// bb0: {
//     var0 = const 0u32;
//     var1 = const true;
//     goto -> bb1;
// }
//
// bb1: {
//     tmp1 = var1;
//     if(tmp1) -> [true: bb3, false: bb2];
// }
//
// bb2: {
//     tmp0 = ();
//     tmp7 = var0;
//     tmp6 = println(tmp7) -> bb7;
// }
//
// bb3: {
//     tmp5 = var0;
//     tmp4 = Ne(tmp5, const 0u32);
//     if(tmp4) -> [true: bb4, false: bb5];
// }
//
// bb4: {
//     var0 = const 1u32;
//     tmp3 = ();
//     goto -> bb6;
// }
//
// bb5: {
//     tmp3 = ();
//     goto -> bb6;
// }
//
// bb6: {
//     var1 = const false;
//     tmp2 = ();
//     goto -> bb1;
// }
// END rustc.node4.ConstPropagate.before.mir
// START rustc.node4.DeadCode.after.mir
// bb0: {
//     var1 = const true;
//     goto -> bb1;
// }
//
// bb1: {
//     tmp1 = var1;
//     if(tmp1) -> [true: bb3, false: bb2];
// }
//
// bb2: {
//     tmp6 = println(const 0u32) -> bb4;
// }
//
// bb3: {
//     var1 = const false;
//     goto -> bb1;
// }
// END rustc.node4.DeadCode.after.mir
