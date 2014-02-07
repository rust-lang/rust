// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-fast check-fast doesn't like aux-build
// aux-build:issue-8044.rs

extern mod minimal = "issue-8044";
use minimal::{BTree, leaf};

pub fn main() {
    BTree::<int> { node: leaf(1) };
}
