// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::result::{Result,Ok};

static C: Result<(), Box<int>> = Ok(());

// This is because of yet another bad assertion (ICE) about the null side of a nullable enum.
// So we won't actually compile if the bug is present, but we check the value in main anyway.

pub fn main() {
    assert!(C.is_ok());
}
