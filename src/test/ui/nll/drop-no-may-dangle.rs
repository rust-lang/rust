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
// in the type of `p` must include everything until `p` is dropped
// because of destructor. (Note that the stderr also identifies this
// destructor in the error message.)

// compile-flags:-Zborrowck=mir

#![allow(warnings)]
#![feature(dropck_eyepatch)]

fn use_x(_: usize) -> bool { true }

fn main() {
    let mut v = [1, 2, 3];
    let p: WrapMayNotDangle<&usize> = WrapMayNotDangle { value: &v[0] };
    if true {
        use_x(*p.value);
    } else {
        use_x(22);
        v[0] += 1; //~ ERROR cannot assign to `v[..]` because it is borrowed
    }

    v[0] += 1; //~ ERROR cannot assign to `v[..]` because it is borrowed
}

struct WrapMayNotDangle<T> {
    value: T
}

impl<T> Drop for WrapMayNotDangle<T> {
    fn drop(&mut self) { }
}
