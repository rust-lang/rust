// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// #47096

#![feature(slice_patterns)]

fn foo(s: &[i32]) -> &[i32] {
    let &[ref xs..] = s;
    xs
}

fn main() {
    let x = [1, 2, 3];
    let y = foo(&x);
    assert_eq!(x, y);
}
