// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z span_free_formats

// Tests that MIR inliner can handle closure arguments. (#45894)

fn main() {
    println!("{}", foo(0, 14));
}

fn foo<T: Copy>(_t: T, q: i32) -> i32 {
    let x = |_t, _q| _t;
    x(q*2, q*3)
}

// END RUST SOURCE
// START rustc.foo.Inline.after.mir
// ...
// bb0: {
//     StorageLive(_3);
//     _3 = [closure@NodeId(28)];
//     StorageLive(_4);
//     _4 = &_3;
//     StorageLive(_5);
//     StorageLive(_6);
//     StorageLive(_7);
//     _7 = _2;
//     _6 = Mul(_7, const 2i32);
//     StorageDead(_7);
//     StorageLive(_8);
//     StorageLive(_9);
//     _9 = _2;
//     _8 = Mul(_9, const 3i32);
//     StorageDead(_9);
//     _5 = (_6, _8);
//     _0 = (_5.0: i32);
//     StorageDead(_5);
//     StorageDead(_8);
//     StorageDead(_6);
//     StorageDead(_4);
//     StorageDead(_3);
//     return;
// }
// ...
// END rustc.foo.Inline.after.mir