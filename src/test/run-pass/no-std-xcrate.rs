// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test #9839
// aux-build:no_std_crate.rs

// This tests that crates which link to std can also be linked to crates with
// #[no_std] that have no lang items.

extern mod no_std_crate;

fn main() {
    no_std_crate::foo();
}
