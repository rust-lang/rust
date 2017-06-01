// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Test that when a crate is linked multiple times that the shortest absolute path name is used

mod foo {
    pub extern crate core;
}

extern crate core;

fn assert_clone<T>() where T : Clone { }

fn main() {
    assert_clone::<foo::core::sync::atomic::AtomicBool>();
    //~^ ERROR `core::sync::atomic::AtomicBool: core::clone::Clone` is not satisfied
}
