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
//
// (This test can be removed entirely when we remove the
// `unsafe_destructor` feature itself.)

struct D<'a>(&'a u32);

#[unsafe_destructor]
//~^ ERROR `#[unsafe_destructor]` does nothing anymore
//~| HELP: add #![feature(unsafe_destructor)] to the crate attributes to enable
// (but of couse there is no point in doing so)
impl<'a> Drop for D<'a> {
    fn drop(&mut self) { }
}

pub fn main() { }
