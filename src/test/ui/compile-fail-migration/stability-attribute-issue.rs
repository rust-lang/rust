// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:stability_attribute_issue.rs
// ignore-tidy-linelength

#![deny(deprecated)]

extern crate stability_attribute_issue;
use stability_attribute_issue::*;

fn main() {
    unstable();
    //~^ ERROR use of unstable library feature 'unstable_test_feature' (see issue #1)
    unstable_msg();
    //~^ ERROR use of unstable library feature 'unstable_test_feature': message (see issue #2)
}
