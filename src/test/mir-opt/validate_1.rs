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

fn foo(_x: &mut i32) {}

fn main() {
    let mut x = 0;
    foo(&mut x);
}

// END RUST SOURCE
// START rustc.node4.EraseRegions.after.mir
// fn foo(_1: &ReErased mut i32) -> () {
//     bb0: {
//         Validate(Acquire, [_1: &ReFree(DefId { krate: CrateNum(0), node: DefIndex(3) => validate_1/8cd878b::foo[0] }, BrAnon(0)) mut i32]);
//         return;
//     }
// }
// END rustc.node4.EraseRegions.after.mir
// START rustc.node11.EraseRegions.after.mir
// fn main() -> () {
//     bb0: {
//         Validate(Suspend(ReScope(Misc(NodeId(20)))), [_1: i32]);
//         _4 = &ReErased mut _1;
//         Validate(Acquire, [(*_4): i32/ReScope(Misc(NodeId(20)))]);
//         Validate(Suspend(ReScope(Misc(NodeId(20)))), [(*_4): i32/ReScope(Misc(NodeId(20)))]);
//         _3 = &ReErased mut (*_4);
//         Validate(Acquire, [(*_3): i32/ReScope(Misc(NodeId(20)))]);
//         Validate(Release, [_3: &ReScope(Misc(NodeId(20))) mut i32]);
//         _2 = const foo(_3) -> bb1;
//     }
//
//     bb1: {
//         Validate(Acquire, [_2: ()]);
//         EndRegion(ReScope(Misc(NodeId(20))));
//         return;
//     }
// }
// END rustc.node11.EraseRegions.after.mir
