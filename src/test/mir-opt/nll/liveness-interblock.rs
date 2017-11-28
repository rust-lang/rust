// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags:-Znll

fn cond() -> bool { false }

fn make_live(_: usize) { }

fn make_dead() { }

fn main() {
    let x = 5;

    if cond() {
        make_live(x);
    } else {
        // x should be dead on entry to this block
        make_dead();
    }
}

// END RUST SOURCE
// START rustc.main.nll.0.mir
//     | Live variables on entry to bb2: [_1]
//     bb2: {
//             | Live variables at bb2[0]: [_1]
//         StorageLive(_4);
//             | Live variables at bb2[1]: [_1]
//         _4 = _1;
//             | Live variables at bb2[2]: [_4]
//         _3 = const make_live(move _4) -> bb4;
//     }
// END rustc.main.nll.0.mir
// START rustc.main.nll.0.mir
//     | Live variables on entry to bb3: []
//     bb3: {
//             | Live variables at bb3[0]: []
//         _5 = const make_dead() -> bb5;
//     }
// END rustc.main.nll.0.mir


