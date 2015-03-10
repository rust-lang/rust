// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that `#[unsafe_destructor]` attribute is gated by `unsafe_destructor`
// feature gate.

struct D<'a>(&'a u32);

#[unsafe_destructor]
impl<'a> Drop for D<'a> {
    //~^ ERROR `#[unsafe_destructor]` allows too many unsafe patterns
    fn drop(&mut self) { }
}
//~^ HELP: add #![feature(unsafe_destructor)] to the crate attributes to enable

pub fn main() { }
