// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

const C1: &'static mut [usize] = &mut [];
//~^ ERROR: references in constants may only refer to immutable values
//~| ERROR: references in constants may only refer to immutable values

static mut S: usize = 3;
const C2: &'static mut usize = unsafe { &mut S };
//~^ ERROR: constants cannot refer to statics
//~| ERROR: references in constants may only refer to immutable values
//~| ERROR: references in constants may only refer to immutable values
//~| ERROR: references in constants may only refer to immutable values

fn main() {}
