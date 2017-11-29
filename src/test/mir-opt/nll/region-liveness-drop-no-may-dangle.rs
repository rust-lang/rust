// Copyright 2012-2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Basic test for liveness constraints: the region (`R1`) that appears
// in the type of `p` includes the points after `&v[0]` up to (but not
// including) the call to `use_x`. The `else` branch is not included.

// ignore-tidy-linelength
// compile-flags:-Znll -Zverbose
//                     ^^^^^^^^^ force compiler to dump more region information

#![allow(warnings)]

fn use_x(_: usize) -> bool { true }

fn main() {
    let mut v = [1, 2, 3];
    let p: Wrap<& /* R1 */ usize> = Wrap { value: &v[0] };
    if true {
        use_x(*p.value);
    } else {
        use_x(22);
    }

    // `p` will get dropped here. Because the `#[may_dangle]`
    // attribute is not present on `Wrap`, we must conservatively
    // assume that the dtor may access the `value` field, and hence we
    // must consider R1 to be live.
}

struct Wrap<T> {
    value: T
}

// Look ma, no `#[may_dangle]` attribute here.
impl<T> Drop for Wrap<T> {
    fn drop(&mut self) { }
}

// END RUST SOURCE
// START rustc.main.nll.0.mir
// | '_#5r: {bb2[3], bb2[4], bb2[5], bb3[0], bb3[1], bb3[2], bb4[0], bb5[0], bb5[1], bb5[2], bb6[0], bb7[0], bb7[1], bb8[0]}
// END rustc.main.nll.0.mir
