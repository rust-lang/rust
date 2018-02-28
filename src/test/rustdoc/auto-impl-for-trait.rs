// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test for https://github.com/rust-lang/rust/issues/48463 issue.

use std::any::Any;
use std::ops::Deref;

pub struct AnyValue {
    val: Box<Any>,
}

impl Deref for AnyValue {
    type Target = Any;

    fn deref(&self) -> &Any {
        &*self.val
    }
}
