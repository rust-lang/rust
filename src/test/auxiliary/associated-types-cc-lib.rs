// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Helper for test issue-18048, which tests associated types in a
// cross-crate scenario.

#![crate_type="lib"]

pub trait Bar: Sized {
    type T;

    fn get(x: Option<Self>) -> <Self as Bar>::T;
}

impl Bar for isize {
    type T = usize;

    fn get(_: Option<isize>) -> usize { 22 }
}
