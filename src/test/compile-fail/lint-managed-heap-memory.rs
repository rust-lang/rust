// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]
#![forbid(managed_heap_memory)]

use std::gc::{Gc, GC};

struct Foo {
    x: Gc<int> //~ ERROR type uses managed
}

fn main() {
    let _x : Foo = Foo {x : box(GC) 10};
    //~^ ERROR type uses managed
}
