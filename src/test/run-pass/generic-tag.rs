// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(managed_boxes)]
#![allow(dead_assignment)]
#![allow(unused_variable)]

use std::gc::{Gc, GC};

enum option<T> { some(Gc<T>), none, }

pub fn main() {
    let mut a: option<int> = some::<int>(box(GC) 10);
    a = none::<int>;
}
