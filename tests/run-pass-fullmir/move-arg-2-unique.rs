// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//ignore-msvc

#![allow(unused_features, unused_variables)]
#![feature(box_syntax)]

fn test(foo: Box<Vec<isize>> ) { assert_eq!((*foo)[0], 10); }

pub fn main() {
    let x = box vec![10];
    // Test forgetting a local by move-in
    test(x);
}
