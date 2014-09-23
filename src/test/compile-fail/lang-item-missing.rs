// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that a missing lang item (in this case `sized`) does not cause an ICE,
// see #17392.

// error-pattern: requires `sized` lang_item

#![no_std]

#[start]
fn start(argc: int, argv: *const *const u8) -> int {
    0
}
