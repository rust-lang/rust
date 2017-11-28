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
    x(q, q)
}

// END RUST SOURCE
// START rustc.foo.Inline.after.mir
// ...
// bb0: {
//     ...
//     _3 = [closure@NodeId(28)];
//     ...
//     _4 = &_3;
//     ...
//     _6 = _2;
//     ...
//     _7 = _2;
//     _5 = (move _6, move _7);
//     _8 = move (_5.0: i32);
//     _9 = move (_5.1: i32);
//     _0 = move _8;
//     ...
//     return;
// }
// ...
// END rustc.foo.Inline.after.mir
