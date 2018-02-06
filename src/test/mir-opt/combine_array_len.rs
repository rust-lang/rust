// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn norm2(x: [f32; 2]) -> f32 {
    let a = x[0];
    let b = x[1];
    a*a + b*b
}

fn main() {
    assert_eq!(norm2([3.0, 4.0]), 5.0*5.0);
}

// END RUST SOURCE

// START rustc.norm2.InstCombine.before.mir
//     _5 = Len(_1);
//     ...
//     _10 = Len(_1);
// END rustc.norm2.InstCombine.before.mir

// START rustc.norm2.InstCombine.after.mir
//     _5 = const 2usize;
//     ...
//     _10 = const 2usize;
// END rustc.norm2.InstCombine.after.mir
