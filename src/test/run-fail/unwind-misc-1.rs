// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern:fail

extern mod std;
use std::map;
use std::map::HashMap;

fn main() {
    let count = @mut 0u;
    let map = map::HashMap();
    let mut arr = ~[];
    for uint::range(0u, 10u) |i| {
        arr += ~[@~"key stuff"];
        map.insert(arr, arr + ~[@~"value stuff"]);
        if arr.len() == 5 {
            fail;
        }
    }
}
