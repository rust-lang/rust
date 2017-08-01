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

// Make sure unsafe fns and fns with an unsafe block only get restricted validation.

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
    test_closure(&mut 0);
}

// FIXME: Also test code generated inside the closure, make sure it only does restricted validation
// because it is entirely inside an unsafe block.  Unfortunately, the interesting lines of code also
// contain name of the source file, so we cannot test for it.

// END RUST SOURCE
// START rustc.node4.EraseRegions.after.mir
// fn write_42(_1: *mut i32) -> bool {
//     bb0: {
//         Validate(Acquire, [_1: *mut i32]);
//         Validate(Release, [_1: *mut i32]);
//         return;
//     }
// }
// END rustc.node4.EraseRegions.after.mir
// START rustc.node17.EraseRegions.after.mir
// fn test(_1: &ReErased mut i32) -> () {
//     bb0: {
//         Validate(Acquire, [_1: &ReFree(DefId { krate: CrateNum(0), node: DefIndex(4) => validate_4/8cd878b::test[0] }, BrAnon(0)) mut i32]);
//         Validate(Release, [_1: &ReFree(DefId { krate: CrateNum(0), node: DefIndex(4) => validate_4/8cd878b::test[0] }, BrAnon(0)) mut i32]);
//         _3 = const write_42(_4) -> bb1;
//     }
//     bb1: {
//         Validate(Acquire, [_3: bool]);
//         Validate(Release, [_3: bool]);
//     }
// }
// END rustc.node17.EraseRegions.after.mir
