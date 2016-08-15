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
    let cfg = cfg!(never_ever_set);
    let x = 42 + if cfg { 42 } else { 0 };
    ::std::process::exit(x - 42);
}

// END RUST SOURCE
// START rustc.node4.ConstPropagate.before.mir
// bb0: {
//     var0 = const false;
//     tmp1 = var0;
//     if(tmp1) -> [true: bb1, false: bb2];
// }
//
// bb1: {
//     tmp0 = const 42i32;
//     goto -> bb3;
// }
//
// bb2: {
//     tmp0 = const 0i32;
//     goto -> bb3;
// }
//
// bb3: {
//     var1 = Add(const 42i32, tmp0);
//     tmp4 = var1;
//     tmp3 = Sub(tmp4, const 42i32);
//     std::process::exit(tmp3);
// }
// END rustc.node4.ConstPropagate.before.mir
// START rustc.node4.DeadCode.after.mir
// bb1: {
//     std::process::exit(const 0i32);
// }
// END rustc.node4.DeadCode.after.mir
