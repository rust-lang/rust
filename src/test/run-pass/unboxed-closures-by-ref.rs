// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test by-ref capture of environment in unboxed closure types

fn call_fn<F: Fn()>(f: F) {
    f()
}

fn call_fn_mut<F: FnMut()>(mut f: F) {
    f()
}

fn call_fn_once<F: FnOnce()>(f: F) {
    f()
}

fn main() {
    let mut x = 0_usize;
    let y = 2_usize;

    call_fn(|| assert_eq!(x, 0));
    call_fn_mut(|| x += y);
    call_fn_once(|| x += y);
    assert_eq!(x, y * 2);
}
