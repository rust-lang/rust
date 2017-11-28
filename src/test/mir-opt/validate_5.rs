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
// compile-flags: -Z verbose -Z mir-emit-validate=2 -Z span_free_formats

// Make sure unsafe fns and fns with an unsafe block still get full validation.

unsafe fn write_42(x: *mut i32) -> bool {
    *x = 42;
    true
}

fn test(x: &mut i32) {
    unsafe { write_42(x) };
}

fn main() {
    test(&mut 0);

    let test_closure = unsafe { |x: &mut i32| write_42(x) };
    // Note that validation will fail if this is executed: The closure keeps the lock on
    // x, so the write in write_42 fails.  This test just checks code generation,
    // so the UB doesn't matter.
    test_closure(&mut 0);
}

// END RUST SOURCE
// START rustc.test.EraseRegions.after.mir
// fn test(_1: &ReErased mut i32) -> () {
//     ...
//     bb0: {
//         Validate(Acquire, [_1: &ReFree(DefId(0/0:4 ~ validate_5[317d]::test[0]), BrAnon(0)) mut i32]);
//         ...
//         Validate(Release, [_2: bool, _3: *mut i32]);
//         _2 = const write_42(move _3) -> bb1;
//     }
//     ...
// }
// END rustc.test.EraseRegions.after.mir
// START rustc.main-{{closure}}.EraseRegions.after.mir
// fn main::{{closure}}(_1: &ReErased [closure@NodeId(46)], _2: &ReErased mut i32) -> bool {
//     ...
//     bb0: {
//         Validate(Acquire, [_1: &ReFree(DefId(0/1:9 ~ validate_5[317d]::main[0]::{{closure}}[0]), BrEnv) [closure@NodeId(46)], _2: &ReFree(DefId(0/1:9 ~ validate_5[317d]::main[0]::{{closure}}[0]), BrAnon(0)) mut i32]);
//         StorageLive(_3);
//         StorageLive(_4);
//         Validate(Suspend(ReScope(Node(ItemLocalId(9)))), [(*_2): i32]);
//         _4 = &ReErased mut (*_2);
//         Validate(Acquire, [(*_4): i32/ReScope(Node(ItemLocalId(9)))]);
//         _3 = move _4 as *mut i32 (Misc);
//         EndRegion(ReScope(Node(ItemLocalId(9))));
//         StorageDead(_4);
//         Validate(Release, [_0: bool, _3: *mut i32]);
//         _0 = const write_42(move _3) -> bb1;
//     }
//     ...
// }
// END rustc.main-{{closure}}.EraseRegions.after.mir
