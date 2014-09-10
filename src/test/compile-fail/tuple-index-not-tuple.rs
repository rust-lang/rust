// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(tuple_indexing)]

struct Point { x: int, y: int }
struct Empty;

fn main() {
    let origin = Point { x: 0, y: 0 };
    origin.0;
    //~^ ERROR attempted tuple index `0` on type `Point`, but the type was not
    Empty.0;
    //~^ ERROR attempted tuple index `0` on type `Empty`, but the type was not
}
