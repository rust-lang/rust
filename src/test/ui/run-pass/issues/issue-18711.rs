// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we don't panic on a RefCell borrow conflict in certain
// code paths involving unboxed closures.

// pretty-expanded FIXME #23616

// aux-build:issue-18711.rs
extern crate issue_18711 as issue;

fn main() {
    (|| issue::inner(()))();
}
