// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:option_helpers.rs

#![warn(clippy::iter_skip_next)]
#![allow(clippy::blacklisted_name)]

extern crate option_helpers;

use option_helpers::IteratorFalsePositives;

/// Checks implementation of `ITER_SKIP_NEXT` lint
fn iter_skip_next() {
    let mut some_vec = vec![0, 1, 2, 3];
    let _ = some_vec.iter().skip(42).next();
    let _ = some_vec.iter().cycle().skip(42).next();
    let _ = (1..10).skip(10).next();
    let _ = &some_vec[..].iter().skip(3).next();
    let foo = IteratorFalsePositives { foo: 0 };
    let _ = foo.skip(42).next();
    let _ = foo.filter().skip(42).next();
}

fn main() {}
