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

fn main() {
    // Make sure the function actually gets instantiated.
    test1(false, 0);
}

// END RUST SOURCE
// START rustc.test1.Deaggregator.before.mir
//  bb1: {
//      StorageLive(_4);
//      _4 = _2;
//      _0 = Foo::A(move _4,);
//      StorageDead(_4);
//      goto -> bb3;
//  }
//  bb2: {
//      StorageLive(_5);
//      _5 = _2;
//      _0 = Foo::B(move _5,);
//      StorageDead(_5);
//      goto -> bb3;
//  }
// END rustc.test1.Deaggregator.before.mir
// START rustc.test1.Deaggregator.after.mir
//  bb1: {
//      StorageLive(_4);
//      _4 = _2;
//      ((_0 as A).0: i32) = move _4;
//      discriminant(_0) = 0;
//      StorageDead(_4);
//      goto -> bb3;
//  }
//  bb2: {
//      StorageLive(_5);
//      _5 = _2;
//      ((_0 as B).0: i32) = move _5;
//      discriminant(_0) = 1;
//      StorageDead(_5);
//      goto -> bb3;
//  }
// END rustc.test1.Deaggregator.after.mir
//
