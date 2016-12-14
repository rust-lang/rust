// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn test(x: u32) -> u32 {
    let y = x;
    y
}

fn main() { }

// END RUST SOURCE
// START rustc.node4.CopyPropagation.before.mir
//  bb0: {
//      _2 = _1;
//      _4 = _2;
//      _3 = _4;
//      _5 = _3;
//      _0 = _5;
//      return;
//  }
// END rustc.node4.CopyPropagation.before.mir
// START rustc.node4.CopyPropagation.after.mir
//  bb0: {
//      _0 = _1;
//      return;
//  }
// END rustc.node4.CopyPropagation.after.mir
