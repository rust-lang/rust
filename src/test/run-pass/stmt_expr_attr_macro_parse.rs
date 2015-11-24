// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

macro_rules! m {
    ($e:expr) => {
        "expr includes attr"
    };
    (#[$attr:meta] $e:expr) => {
        "expr excludes attr"
    }
}

macro_rules! n {
    (#[$attr:meta] $e:expr) => {
        "expr excludes attr"
    };
    ($e:expr) => {
        "expr includes attr"
    }
}

fn main() {
    assert_eq!(m!(#[attr] 1), "expr includes attr");
    assert_eq!(n!(#[attr] 1), "expr excludes attr");
}
