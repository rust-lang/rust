// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Checks that immutable static items can't have mutable slices

static TEST: &'static mut [isize] = &mut [];
//~^ ERROR statics are not allowed to have mutable references

pub fn main() { }
