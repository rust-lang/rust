// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-31702-1.rs
// aux-build:issue-31702-2.rs
// ignore-test: FIXME(#31702) when this test was added it was thought that the
//                            accompanying llvm update would fix it, but
//                            unfortunately it appears that was not the case. In
//                            the interest of not deleting the test, though,
//                            this is just tagged with ignore-test

// this test is actually entirely in the linked library crates

extern crate issue_31702_1;
extern crate issue_31702_2;

fn main() {}
