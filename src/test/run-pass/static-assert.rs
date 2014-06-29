// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[static_assert]
static b: bool = true;

#[static_assert]
static c: bool = 1i == 1;

#[static_assert]
static d: bool = 1i != 2;

#[static_assert]
static f: bool = (4i/2) == 2;

pub fn main() {
}
