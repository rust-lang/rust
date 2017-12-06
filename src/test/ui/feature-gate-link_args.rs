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
// feature gate, both when it occurs where expected (atop
// `extern { }` blocks) and where unexpected.

// sidestep warning (which is correct, but misleading for
// purposes of this test)
#![allow(unused_attributes)]

#![link_args = "-l unexpected_use_as_inner_attr_on_mod"]
//~^ ERROR the `link_args` attribute is experimental

#[link_args = "-l expected_use_case"]
//~^ ERROR the `link_args` attribute is experimental
extern {}

#[link_args = "-l unexected_use_on_non_extern_item"]
//~^ ERROR: the `link_args` attribute is experimental
fn main() {}
