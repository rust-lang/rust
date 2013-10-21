// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[deriving(Clone)]
struct Foo {
    a: ~str,
}

pub fn main() {
    let x = [ @[true], ..512 ];
    let y = [ 0, ..1 ];

    error!("{:?}", x);
    error!("{:?}", y);
}
