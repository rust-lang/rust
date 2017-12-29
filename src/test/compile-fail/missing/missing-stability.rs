// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Checks that exported items without stability attributes cause an error

#![crate_type="lib"]
#![feature(staged_api)]

#![stable(feature = "test_feature", since = "1.0.0")]

pub fn unmarked() {
    //~^ ERROR This node does not have a stability attribute
    ()
}

#[unstable(feature = "foo", issue = "0")]
pub mod foo {
    // #[unstable] is inherited
    pub fn unmarked() {}
}

#[stable(feature = "bar", since="1.0.0")]
pub mod bar {
    // #[stable] is not inherited
    pub fn unmarked() {}
    //~^ ERROR This node does not have a stability attribute
}
