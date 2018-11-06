// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z span_free_formats -Z mir-emit-retag

// Tests that MIR inliner fixes up `Retag`'s `fn_entry` flag

fn main() {
    println!("{}", bar());
}

#[inline(always)]
fn foo(x: &i32, y: &i32) -> bool {
    *x == *y
}

fn bar() -> bool {
    let f = foo;
    f(&1, &-1)
}

// END RUST SOURCE
// START rustc.bar.Inline.after.mir
// ...
//     bb0: {
//         ...
//         Retag(_3);
//         ...
//         Retag(_3);
//         Retag(_6);
//         StorageLive(_9);
//         _9 = (*_3);
//         StorageLive(_10);
//         _10 = (*_6);
//         _0 = Eq(move _9, move _10);
//         ...
//         return;
//     }
// ...
// END rustc.bar.Inline.after.mir
