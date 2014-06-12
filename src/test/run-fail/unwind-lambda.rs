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

#![feature(managed_boxes)]

use std::gc::{Gc, GC};

fn main() {
    let cheese = "roquefort".to_string();
    let carrots = box(GC) "crunchy".to_string();

    let result: |Gc<String>, |String||: 'static = (|tasties, macerate| {
        macerate((*tasties).clone());
    });
    result(carrots, |food| {
        let mush = format!("{}{}", food, cheese);
        let cheese = cheese.clone();
        let f: || = || {
            let _chew = format!("{}{}", mush, cheese);
            fail!("so yummy")
        };
        f();
    });
}
