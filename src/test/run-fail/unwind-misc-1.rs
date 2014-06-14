// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// exec-env:RUST_NEWRT=1
// error-pattern:fail

#![feature(managed_boxes)]

use std::vec;
use std::collections;
use std::gc::GC;

fn main() {
    let _count = box(GC) 0u;
    let mut map = collections::HashMap::new();
    let mut arr = Vec::new();
    for _i in range(0u, 10u) {
        arr.push(box(GC) "key stuff".to_string());
        map.insert(arr.clone(),
                   arr.clone().append([box(GC) "value stuff".to_string()]));
        if arr.len() == 5 {
            fail!();
        }
    }
}
