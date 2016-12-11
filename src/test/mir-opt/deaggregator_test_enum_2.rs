// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that deaggregate fires in more than one basic block

enum Foo {
    A(i32),
    B(i32),
}

fn test1(x: bool, y: i32) -> Foo {
    if x {
        Foo::A(y)
    } else {
        Foo::B(y)
    }
}

fn main() {}

// END RUST SOURCE
// START rustc.node12.Deaggregator.before.mir
//  bb1: {
//      _6 = _4;
//      _0 = Foo::A(_6,);
//      goto -> bb3;
//  }
//
//  bb2: {
//      _7 = _4;
//      _0 = Foo::B(_7,);
//      goto -> bb3;
//  }
// END rustc.node12.Deaggregator.before.mir
// START rustc.node12.Deaggregator.after.mir
//  bb1: {
//      _6 = _4;
//      ((_0 as A).0: i32) = _6;
//      discriminant(_0) = 0;
//      goto -> bb3;
//  }
//
//  bb2: {
//      _7 = _4;
//      ((_0 as B).0: i32) = _7;
//      discriminant(_0) = 1;
//      goto -> bb3;
//  }
// END rustc.node12.Deaggregator.after.mir
//
