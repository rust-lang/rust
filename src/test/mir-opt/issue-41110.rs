// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-wasm32-bare compiled with panic=abort by default

// check that we don't emit multiple drop flags when they are not needed.

fn main() {
    let x = S.other(S.id());
}

// no_mangle to make sure this gets instantiated even in an executable.
#[no_mangle]
pub fn test() {
    let u = S;
    let mut v = S;
    drop(v);
    v = u;
}

struct S;
impl Drop for S {
    fn drop(&mut self) {
    }
}

impl S {
    fn id(self) -> Self { self }
    fn other(self, s: Self) {}
}

// END RUST SOURCE
// START rustc.main.ElaborateDrops.after.mir
//    let mut _0: ();
//    scope 1 {
//        let _1: ();
//    }
//    ...
//    let mut _2: S;
//    let mut _3: S;
//    let mut _4: S;
//    let mut _5: bool;
//    bb0: {
// END rustc.main.ElaborateDrops.after.mir
// START rustc.test.ElaborateDrops.after.mir
//    let mut _0: ();
//    ...
//    let _1: S;
//    ...
//    let mut _2: S;
//    ...
//    let mut _3: ();
//    let mut _4: S;
//    let mut _5: S;
//    let mut _6: bool;
//    bb0: {
// END rustc.test.ElaborateDrops.after.mir
