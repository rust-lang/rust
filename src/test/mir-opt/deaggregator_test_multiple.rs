// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that deaggregate fires more than once per block

enum Foo {
    A(i32),
    B,
}

fn test(x: i32) -> [Foo; 2] {
    [Foo::A(x), Foo::A(x)]
}

fn main() { }

// END RUST SOURCE
// START rustc.node10.Deaggregator.before.mir
// bb0: {
//     _2 = _1;
//     _4 = _2;
//     _3 = Foo::A(_4,);
//     _6 = _2;
//     _5 = Foo::A(_6,);
//     _0 = [_3, _5];
//     return;
// }
// END rustc.node10.Deaggregator.before.mir
// START rustc.node10.Deaggregator.after.mir
// bb0: {
//     _2 = _1;
//     _4 = _2;
//     ((_3 as A).0: i32) = _4;
//     discriminant(_3) = 0;
//     _6 = _2;
//     ((_5 as A).0: i32) = _6;
//     discriminant(_5) = 0;
//     _0 = [_3, _5];
//     return;
// }
// END rustc.node10.Deaggregator.after.mir
