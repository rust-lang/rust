// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that `#[link_args]` attribute is gated by `link_args`
// feature gate.

#[link_args = "aFdEfSeVEEE"]
extern {}
//~^ ERROR the `link_args` attribute is not portable across platforms
//~| HELP add #![feature(link_args)] to the crate attributes to enable

fn main() { }
