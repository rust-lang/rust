// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength
// compile-flags: -Z verbose -Z mir-emit-validate=1

struct Test;

impl Test {
    // Make sure we run the pass on a method, not just on bare functions.
    fn foo(&self, _x: &mut i32) {}
}

fn main() {
    let mut x = 0;
    Test.foo(&mut x);
}

// END RUST SOURCE
// START rustc.node10.EraseRegions.after.mir
//     bb0: {
//         Validate(Acquire, [_1: &ReFree(DefId { krate: CrateNum(0), node: DefIndex(5) => validate_1/8cd878b::{{impl}}[0]::foo[0] }, BrAnon(0)) Test, _2: &ReFree(DefId { krate: CrateNum(0), node: DefIndex(5) => validate_1/8cd878b::{{impl}}[0]::foo[0] }, BrAnon(1)) mut i32]);
//         return;
//     }
// END rustc.node10.EraseRegions.after.mir
// START rustc.node21.EraseRegions.after.mir
// fn main() -> () {
//     bb0: {
//         Validate(Suspend(ReScope(Misc(NodeId(30)))), [_1: i32]);
//         _6 = &ReErased mut _1;
//         Validate(Acquire, [(*_6): i32/ReScope(Misc(NodeId(30)))]);
//         Validate(Suspend(ReScope(Misc(NodeId(30)))), [(*_6): i32/ReScope(Misc(NodeId(30)))]);
//         _5 = &ReErased mut (*_6);
//         Validate(Acquire, [(*_5): i32/ReScope(Misc(NodeId(30)))]);
//         Validate(Release, [_3: &ReScope(Misc(NodeId(30))) Test, _5: &ReScope(Misc(NodeId(30))) mut i32]);
//         _2 = const Test::foo(_3, _5) -> bb1;
//     }
//
//     bb1: {
//         Validate(Acquire, [_2: ()]);
//         EndRegion(ReScope(Misc(NodeId(30))));
//         return;
//     }
// }
// END rustc.node21.EraseRegions.after.mir
