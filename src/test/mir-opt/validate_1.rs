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
// compile-flags: -Z verbose -Z mir-emit-validate=1 -Z span_free_formats

struct Test(i32);

impl Test {
    // Make sure we run the pass on a method, not just on bare functions.
    fn foo(&self, _x: &mut i32) {}
}

fn main() {
    let mut x = 0;
    Test(0).foo(&mut x); // just making sure we do not panic when there is a tuple struct ctor

    // Also test closures
    let c = |x: &mut i32| { let y = &*x; *y };
    c(&mut x);
}

// END RUST SOURCE
// START rustc.{{impl}}-foo.EraseRegions.after.mir
//     bb0: {
//         Validate(Acquire, [_1: &ReFree(DefId(0/0:5 ~ validate_1[317d]::{{impl}}[0]::foo[0]), BrAnon(0)) Test, _2: &ReFree(DefId(0/0:5 ~ validate_1[317d]::{{impl}}[0]::foo[0]), BrAnon(1)) mut i32]);
//         ...
//         return;
//     }
// END rustc.{{impl}}-foo.EraseRegions.after.mir
// START rustc.main.EraseRegions.after.mir
// fn main() -> () {
//     ...
//     bb0: {
//         ...
//         Validate(Suspend(ReScope(Node(ItemLocalId(10)))), [_1: i32]);
//         _6 = &ReErased mut _1;
//         Validate(Acquire, [(*_6): i32/ReScope(Node(ItemLocalId(10)))]);
//         Validate(Suspend(ReScope(Node(ItemLocalId(10)))), [(*_6): i32/ReScope(Node(ItemLocalId(10)))]);
//         _5 = &ReErased mut (*_6);
//         Validate(Acquire, [(*_5): i32/ReScope(Node(ItemLocalId(10)))]);
//         Validate(Release, [_2: (), _3: &ReScope(Node(ItemLocalId(10))) Test, _5: &ReScope(Node(ItemLocalId(10))) mut i32]);
//         _2 = const Test::foo(move _3, move _5) -> bb1;
//     }
//
//     bb1: {
//         Validate(Acquire, [_2: ()]);
//         EndRegion(ReScope(Node(ItemLocalId(10))));
//         ...
//         return;
//     }
// }
// END rustc.main.EraseRegions.after.mir
// START rustc.main-{{closure}}.EraseRegions.after.mir
// fn main::{{closure}}(_1: &ReErased [closure@NodeId(50)], _2: &ReErased mut i32) -> i32 {
//     ...
//     bb0: {
//         Validate(Acquire, [_1: &ReFree(DefId(0/1:11 ~ validate_1[317d]::main[0]::{{closure}}[0]), BrEnv) [closure@NodeId(50)], _2: &ReFree(DefId(0/1:11 ~ validate_1[317d]::main[0]::{{closure}}[0]), BrAnon(0)) mut i32]);
//         StorageLive(_3);
//         Validate(Suspend(ReScope(Remainder(BlockRemainder { block: ItemLocalId(22), first_statement_index: 0 }))), [(*_2): i32]);
//         _3 = &ReErased (*_2);
//         Validate(Acquire, [(*_3): i32/ReScope(Remainder(BlockRemainder { block: ItemLocalId(22), first_statement_index: 0 })) (imm)]);
//         StorageLive(_4);
//         _4 = (*_3);
//         _0 = move _4;
//         StorageDead(_4);
//         EndRegion(ReScope(Remainder(BlockRemainder { block: ItemLocalId(22), first_statement_index: 0 })));
//         StorageDead(_3);
//         return;
//     }
// }
// END rustc.main-{{closure}}.EraseRegions.after.mir
