// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::collections::HashMap;

fn copy<T: Copy>(&x: &T) -> T {
    x
}

fn main() {
    let arr = [(1, 1_usize), (2, 2), (3, 3)];

    let v1: Vec<&_> = arr.iter().collect();
    let v2: Vec<_> = arr.iter().map(copy).collect();

    let m1: HashMap<_, _> = arr.iter().map(copy).collect();
    let m2: HashMap<int, _> = arr.iter().map(copy).collect();
    let m3: HashMap<_, uint> = arr.iter().map(copy).collect();
}
