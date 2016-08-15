// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
fn main() {
    let x = 0;
    match x {
        y@0 => ::std::process::exit(y),
        y@1 => ::std::process::exit(y),
        y@2 => ::std::process::exit(y),
        y@3 => ::std::process::exit(y),
        y@_ => ::std::process::exit(y),
    }
}
// END RUST SOURCE
// START rustc.node4.ConstPropagate.before.mir
// bb0: {
//     StorageLive(var0);
//     var0 = const 0i32;
//     switchInt(var0) -> [0i32: bb2, 1i32: bb3, 2i32: bb4, 3i32: bb5, otherwise: bb1];
// }
//
// bb1: {
//     StorageLive(var5);
//     var5 = var0;
//     StorageLive(tmp4);
//     tmp4 = var5;
//     std::process::exit(tmp4);
// }
//
// bb2: {
//     StorageLive(var1);
//     var1 = var0;
//     StorageLive(tmp0);
//     tmp0 = var1;
//     std::process::exit(tmp0);
// }
//
// bb3: {
//     StorageLive(var2);
//     var2 = var0;
//     StorageLive(tmp1);
//     tmp1 = var2;
//     std::process::exit(tmp1);
// }
//
// bb4: {
//     StorageLive(var3);
//     var3 = var0;
//     StorageLive(tmp2);
//     tmp2 = var3;
//     std::process::exit(tmp2);
// }
//
// bb5: {
//     StorageLive(var4);
//     var4 = var0;
//     StorageLive(tmp3);
//     tmp3 = var4;
//     std::process::exit(tmp3);
// }
// END rustc.node4.ConstPropagate.before.mir
// START rustc.node4.DeadCode.after.mir
// bb0: {
//     StorageLive(var0);
//     goto -> bb1;
// }
//
// bb1: {
//     StorageLive(var1);
//     StorageLive(tmp0);
//     std::process::exit(const 0i32);
// }
// END rustc.node4.DeadCode.after.mir
