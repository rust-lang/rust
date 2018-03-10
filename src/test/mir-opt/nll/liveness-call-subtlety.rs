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

fn can_panic() -> Box<usize> {
    Box::new(44)
}

fn main() {
    let mut x = Box::new(22);
    x = can_panic();
}

// Check that:
// - `_1` is the variable corresponding to `x`
// and
// - `_1` is live when `can_panic` is called (because it may be dropped)
//
// END RUST SOURCE
// START rustc.main.nll.0.mir
//    | Live variables on entry to bb0: []
//    bb0: {
//            | Live variables on entry to bb0[0]: []
//        StorageLive(_1);
//            | Live variables on entry to bb0[1]: []
//        _1 = const <std::boxed::Box<T>>::new(const 22usize) -> [return: bb2, unwind: bb1];
//    }
// END rustc.main.nll.0.mir
// START rustc.main.nll.0.mir
//    | Live variables on entry to bb2: [_1 (drop)]
//    bb2: {
//            | Live variables on entry to bb2[0]: [_1 (drop)]
//        StorageLive(_2);
//            | Live variables on entry to bb2[1]: [_1 (drop)]
//        _2 = const can_panic() -> [return: bb3, unwind: bb4];
//    }
// END rustc.main.nll.0.mir
