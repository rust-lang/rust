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

fn main() {
    // Make sure the function actually gets instantiated.
    test(0);
}

// END RUST SOURCE
// START rustc.test.Deaggregator.before.mir
// bb0: {
//     ...
//     _3 = _1;
//     ...
//     _2 = Foo::A(move _3,);
//     ...
//     _5 = _1;
//     _4 = Foo::A(move _5,);
//     ...
//     _0 = [move _2, move _4];
//     ...
//     return;
// }
// END rustc.test.Deaggregator.before.mir
// START rustc.test.Deaggregator.after.mir
// bb0: {
//     ...
//     _3 = _1;
//     ...
//     ((_2 as A).0: i32) = move _3;
//     discriminant(_2) = 0;
//     ...
//     _5 = _1;
//     ((_4 as A).0: i32) = move _5;
//     discriminant(_4) = 0;
//     ...
//     _0 = [move _2, move _4];
//     ...
//     return;
// }
// END rustc.test.Deaggregator.after.mir
