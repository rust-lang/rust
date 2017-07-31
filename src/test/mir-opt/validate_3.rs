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

struct Test {
    x: i32
}

fn foo(_x: &i32) {}

fn main() {
    // These internal unsafe functions should have no effect on the code generation.
    unsafe fn _unused1() {}
    fn _unused2(x: *const i32) -> i32 { unsafe { *x }}

    let t = Test { x: 0 };
    let t = &t;
    foo(&t.x);
}

// END RUST SOURCE
// START rustc.node16.EraseRegions.after.mir
// fn main() -> () {
//     let mut _5: &ReErased i32;
//     bb0: {
//         Validate(Suspend(ReScope(Misc(NodeId(46)))), [((*_2).0: i32): i32/ReScope(Remainder(BlockRemainder { block: NodeId(18), first_statement_index: 3 })) (imm)]);
//         _5 = &ReErased ((*_2).0: i32);
//         Validate(Acquire, [(*_5): i32/ReScope(Misc(NodeId(46))) (imm)]);
//         Validate(Suspend(ReScope(Misc(NodeId(46)))), [(*_5): i32/ReScope(Misc(NodeId(46))) (imm)]);
//         _4 = &ReErased (*_5);
//         Validate(Acquire, [(*_4): i32/ReScope(Misc(NodeId(46))) (imm)]);
//         Validate(Release, [_4: &ReScope(Misc(NodeId(46))) i32]);
//         _3 = const foo(_4) -> bb1;
//     }
//     bb1: {
//         EndRegion(ReScope(Misc(NodeId(46))));
//         EndRegion(ReScope(Remainder(BlockRemainder { block: NodeId(18), first_statement_index: 3 })));
//         return;
//     }
// }
// END rustc.node16.EraseRegions.after.mir
