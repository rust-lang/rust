// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::sys;

pub fn main() {
    unsafe {
        let i = ~@1;
        let j = ~@2;
        let rc1 = sys::refcount(*i);
        let j = i.clone();
        let rc2 = sys::refcount(*i);
        error!("rc1: %u rc2: %u", rc1, rc2);
        assert_eq!(rc1 + 1u, rc2);
    }
}
