// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[cfg(target_pointer_width = "32")]
const EXPECTED_PTR_WIDTH: u32 = 32;
#[cfg(target_pointer_width = "64")]
const EXPECTED_PTR_WIDTH: u32 = 64;

#[cfg(target_pointer_width = "32")]
const EXPECTED_PTR_WIDTH_STR: &'static str = "32";
#[cfg(target_pointer_width = "64")]
const EXPECTED_PTR_WIDTH_STR: &'static str = "64";

fn main() {
    let ptr_width_int = cfg_int!(target_pointer_width);
    let ptr_width_str = cfg_str!(target_pointer_width);
    let ptr_width_float = cfg_float!(target_pointer_width);

    assert_eq!(ptr_width_int, EXPECTED_PTR_WIDTH);
    assert_eq!(ptr_width_str, EXPECTED_PTR_WIDTH_STR);
    assert_eq!(ptr_width_float as u32, EXPECTED_PTR_WIDTH);
}
