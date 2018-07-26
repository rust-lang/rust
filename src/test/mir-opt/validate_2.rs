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
// ignore-wasm32-bare unwinding being disabled causes differences in output
// ignore-wasm64-bare unwinding being disabled causes differences in output
// compile-flags: -Z verbose -Z mir-emit-validate=1

fn main() {
    let _x : Box<[i32]> = Box::new([1, 2, 3]);
}

// END RUST SOURCE
// START rustc.main.EraseRegions.after.mir
// fn main() -> () {
//     ...
//     bb1: {
//         Validate(Acquire, [_2: std::boxed::Box<[i32; 3]>]);
//         Validate(Release, [_2: std::boxed::Box<[i32; 3]>]);
//         _1 = move _2 as std::boxed::Box<[i32]> (Unsize);
//         Validate(Acquire, [_1: std::boxed::Box<[i32]>]);
//         StorageDead(_2);
//         StorageDead(_3);
//         _0 = ();
//         Validate(Release, [_1: std::boxed::Box<[i32]>]);
//         drop(_1) -> [return: bb2, unwind: bb3];
//     }
//     ...
// }
// END rustc.main.EraseRegions.after.mir
