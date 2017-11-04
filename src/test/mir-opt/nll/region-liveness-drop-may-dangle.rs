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

// compile-flags:-Znll -Zverbose
//                     ^^^^^^^^^ force compiler to dump more region information

#![allow(warnings)]
#![feature(dropck_eyepatch)]
#![feature(generic_param_attrs)]

fn use_x(_: usize) -> bool { true }

fn main() {
    let mut v = [1, 2, 3];
    let p: Wrap<& /* R4 */ usize> = Wrap { value: &v[0] };
    if true {
        use_x(*p.value);
    } else {
        use_x(22);
    }

    // `p` will get dropped here. However, because of the
    // `#[may_dangle]` attribute, we do not need to consider R4 live.
}

struct Wrap<T> {
    value: T
}

unsafe impl<#[may_dangle] T> Drop for Wrap<T> {
    fn drop(&mut self) { }
}

// END RUST SOURCE
// START rustc.main.nll.0.mir
// | '_#4r: {bb1[3], bb1[4], bb1[5], bb2[0], bb2[1]}
// END rustc.main.nll.0.mir
