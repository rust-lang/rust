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
// START rustc.node12.nll.0.mir
//    | Variables live on entry to the block bb0:
//    bb0: {
//        | Live variables here: []
//        StorageLive(_1);
//        | Live variables here: []
//        _1 = const <std::boxed::Box<T>>::new(const 22usize) -> bb1;
//    }
// END rustc.node12.nll.0.mir
// START rustc.node12.nll.0.mir
//    | Variables live on entry to the block bb1:
//    | - _1
//    bb1: {
//        | Live variables here: [_1]
//        StorageLive(_2);
//        | Live variables here: [_1]
//        _2 = const can_panic() -> [return: bb2, unwind: bb4];
//    }
// END rustc.node12.nll.0.mir
