// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z identify_regions

// Tests to make sure we correctly generate falseUnwind edges in loops

fn main() {
    // Exit early at runtime. Since only care about the generated MIR
    // and not the runtime behavior (which is exercised by other tests)
    // we just bail early. Without this the test just loops infinitely.
    if true {
        return;
    }
    loop {
        let x = 1;
        continue;
    }
}

// END RUST SOURCE
// START rustc.main.SimplifyCfg-qualify-consts.after.mir
//    ...
//    bb1: { // The cleanup block
//        resume;
//    }
//    ...
//    bb3: { // Entry into the loop
//        _1 = ();
//        goto -> bb4;
//    }
//    bb4: { // The loop_block
//        falseUnwind -> [real: bb5, cleanup: bb1];
//    }
//    bb5: { // The loop body (body_block)
//        StorageLive(_5);
//        _5 = const 1i32;
//        FakeRead(ForLet, _5);
//        StorageDead(_5);
//        goto -> bb4;
//    }
//    ...
// END rustc.main.SimplifyCfg-qualify-consts.after.mir
