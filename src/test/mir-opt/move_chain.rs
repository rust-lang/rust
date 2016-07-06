// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub fn test(a: &[u64; 8]) -> [u64; 8] {
    let b = *a;
    let c = b;
    let d = c;
    d
}

fn main() {}
// END RUST SOURCE
// START rustc.node4.MoveUpPropagation.before.mir
// bb0: {
//     var0 = arg0;                     // scope 0 at main.rs:1:13: 1:14
//     tmp0 = (*var0);                  // scope 1 at main.rs:2:13: 2:15
//     var1 = tmp0;                     // scope 1 at main.rs:2:13: 2:15
//     tmp1 = var1;                     // scope 2 at main.rs:3:13: 3:14
//     var2 = tmp1;                     // scope 2 at main.rs:3:13: 3:14
//     tmp2 = var2;                     // scope 3 at main.rs:4:13: 4:14
//     var3 = tmp2;                     // scope 3 at main.rs:4:13: 4:14
//     tmp3 = var3;                     // scope 4 at main.rs:5:5: 5:6
//     return = tmp3;                   // scope 4 at main.rs:5:5: 5:6
//     goto -> bb1;                     // scope 1 at main.rs:1:1: 6:2
// }
// END rustc.node4.MoveUpPropagation.before.mir
// START rustc.node4.MoveUpPropagation.after.mir
// bb0: {
//     var0 = arg0;                     // scope 0 at main.rs:1:13: 1:14
//     return = (*var0);                // scope 1 at main.rs:2:13: 2:15
//     goto -> bb1;                     // scope 1 at main.rs:1:1: 6:2
// }
// END rustc.node4.MoveUpPropagation.after.mir
