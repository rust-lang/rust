// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//buggy.rs

#![feature(box_syntax)]

use std::collections::HashMap;

fn main() {
    let tmp: Box<_>;
    let mut buggy_map: HashMap<usize, &usize> = HashMap::new();
    // FIXME (#22405): Replace `Box::new` with `box` here when/if possible.
    buggy_map.insert(42, &*Box::new(1)); //~ ERROR borrowed value does not live long enough

    // but it is ok if we use a temporary
    tmp = box 2;
    buggy_map.insert(43, &*tmp);
}
