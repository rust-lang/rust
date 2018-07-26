// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Test that absolute path names are correct when a crate is not linked into the root namespace

// aux-build:issue_1920.rs

mod foo {
    pub extern crate issue_1920;
}

fn assert_clone<T>() where T : Clone { }

fn main() {
    assert_clone::<foo::issue_1920::S>();
    //~^ ERROR `foo::issue_1920::S: std::clone::Clone` is not satisfied
}
