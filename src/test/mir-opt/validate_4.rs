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

// Make sure unsafe fns and fns with an unsafe block only get restricted validation.

unsafe fn write_42(x: *mut i32) -> bool {
    let test_closure = |x: *mut i32| *x = 23;
    test_closure(x);
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
// START rustc.write_42.EraseRegions.after.mir
// fn write_42(_1: *mut i32) -> bool {
//     ...
//     bb0: {
//         Validate(Acquire, [_1: *mut i32]);
//         Validate(Release, [_1: *mut i32]);
//         ...
//         return;
//     }
// }
// END rustc.write_42.EraseRegions.after.mir
// START rustc.write_42-{{closure}}.EraseRegions.after.mir
// fn write_42::{{closure}}(_1: &ReErased [closure@NodeId(22)], _2: *mut i32) -> () {
//     ...
//     bb0: {
//         Validate(Acquire, [_1: &ReFree(DefId(0/1:9 ~ validate_4[317d]::write_42[0]::{{closure}}[0]), BrEnv) [closure@NodeId(22)], _2: *mut i32]);
//         Validate(Release, [_1: &ReFree(DefId(0/1:9 ~ validate_4[317d]::write_42[0]::{{closure}}[0]), BrEnv) [closure@NodeId(22)], _2: *mut i32]);
//         (*_2) = const 23i32;
//         _0 = ();
//         return;
//     }
// }
// END rustc.write_42-{{closure}}.EraseRegions.after.mir
// START rustc.test.EraseRegions.after.mir
// fn test(_1: &ReErased mut i32) -> () {
//     ...
//     bb0: {
//         Validate(Acquire, [_1: &ReFree(DefId(0/0:4 ~ validate_4[317d]::test[0]), BrAnon(0)) mut i32]);
//         Validate(Release, [_1: &ReFree(DefId(0/0:4 ~ validate_4[317d]::test[0]), BrAnon(0)) mut i32]);
//         ...
//         _2 = const write_42(move _3) -> bb1;
//     }
//     bb1: {
//         Validate(Acquire, [_2: bool]);
//         Validate(Release, [_2: bool]);
//         ...
//     }
// }
// END rustc.test.EraseRegions.after.mir
// START rustc.main-{{closure}}.EraseRegions.after.mir
// fn main::{{closure}}(_1: &ReErased [closure@NodeId(60)], _2: &ReErased mut i32) -> bool {
//     ...
//     bb0: {
//         Validate(Acquire, [_1: &ReFree(DefId(0/1:10 ~ validate_4[317d]::main[0]::{{closure}}[0]), BrEnv) [closure@NodeId(60)], _2: &ReFree(DefId(0/1:10 ~ validate_4[317d]::main[0]::{{closure}}[0]), BrAnon(0)) mut i32]);
//         Validate(Release, [_1: &ReFree(DefId(0/1:10 ~ validate_4[317d]::main[0]::{{closure}}[0]), BrEnv) [closure@NodeId(60)], _2: &ReFree(DefId(0/1:10 ~ validate_4[317d]::main[0]::{{closure}}[0]), BrAnon(0)) mut i32]);
//         StorageLive(_3);
//         ...
//         _0 = const write_42(move _3) -> bb1;
//     }
//     ...
// }
// END rustc.main-{{closure}}.EraseRegions.after.mir
