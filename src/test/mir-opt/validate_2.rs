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

fn main() {
    let _x : Box<[i32]> = Box::new([1, 2, 3]);
}

// END RUST SOURCE
// START rustc.node4.EraseRegions.after.mir
// fn main() -> () {
//     bb1: {
//         Validate(Release, [_2: std::boxed::Box<[i32; 3]>]);
//         _1 = _2 as std::boxed::Box<[i32]> (Unsize);
//         Validate(Acquire, [_1: std::boxed::Box<[i32]>]);
//     }
// }
// END rustc.node4.EraseRegions.after.mir
