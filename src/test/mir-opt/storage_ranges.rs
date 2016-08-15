// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength

fn main() {
    let a = 0;
    {
        let b = &Some(a);
    }
    let c = 1;
}

// END RUST SOURCE
// START rustc.node4.PreTrans.after.mir
// bb0: {
//     StorageLive(var0);
//     StorageLive(var1);
//     StorageLive(tmp1);
//     StorageLive(tmp2);
//     StorageDead(tmp2);
//     StorageDead(tmp1);
//     StorageDead(var1);
//     StorageLive(var2);
//     return = ();
//     StorageDead(var2);
//     StorageDead(var0);
//     goto -> bb1;
// }
// END rustc.node4.PreTrans.after.mir
