// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test
fn find<T>(_f: fn(@T) -> bool, _v: [@T]) {}

fn main() {
    let x = 10, arr = [];
    find({|f| f.id == x}, arr);
    arr += [{id: 20}]; // This assigns a type to arr
}
