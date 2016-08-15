// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
fn hdrty() -> usize { 6 }

fn inc(idx: u16) -> usize { 1 }

fn main() {
    let hdrty = hdrty();
    let max = match hdrty {
        0 => 6,
        1 => 2,
        _ => 0,
    };
    let mut i = 0;
    while i < max {
        let next = inc(i as u16);
        i = next;
    }
}

// END RUST SOURCE
// START rustc.node17.ConstPropagate.before.mir
// bb6: {
//     tmp1 = var2;
//     tmp2 = var1;
//     tmp0 = Lt(tmp1, tmp2);
//     if(tmp0) -> [true: bb8, false: bb7];
// }
//
// bb7: {
//     return = ();
//     return;
// }
//
// bb8: {
//     tmp5 = var2;
//     tmp4 = tmp5 as u16 (Misc);
//     var3 = inc(tmp4) -> bb9;
// }
//
// bb9: {
//     tmp6 = var3;
//     var2 = tmp6;
//     tmp3 = ();
//     goto -> bb6;
// }
// END rustc.node17.ConstPropagate.before.mir
// START rustc.node17.CsPropagate.after.mir
// bb6: {
//     tmp1 = var2;
//     tmp2 = var1;
//     tmp0 = Lt(tmp1, tmp2);
//     if(tmp0) -> [true: bb8, false: bb7];
// }
//
// bb7: {
//     return = ();
//     return;
// }
//
// bb8: {
//     tmp5 = var2;
//     tmp4 = tmp5 as u16 (Misc);
//     var3 = inc(tmp4) -> bb9;
// }
//
// bb9: {
//     tmp6 = var3;
//     var2 = tmp6;
//     tmp3 = ();
//     goto -> bb6;
// }
// END rustc.node17.CsPropagate.after.mir
