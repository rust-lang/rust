// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(unboxed_closures)]

fn f<F: Fn<usize>>(_: F) {}
fn main() {
    [1, 2, 3].sort_by(|| panic!());
    [1, 2, 3].sort_by(|tuple| panic!());
    [1, 2, 3].sort_by(|(tuple, tuple2)| panic!());
    f(|| panic!());

    let _it = vec![1, 2, 3].into_iter().enumerate().map(|i, x| i);
    let _it = vec![1, 2, 3].into_iter().enumerate().map(|i: usize, x| i);
    let _it = vec![1, 2, 3].into_iter().enumerate().map(|i, x, y| i);
}
