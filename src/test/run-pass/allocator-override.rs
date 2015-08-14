// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// no-prefer-dynamic
// aux-build:allocator-dummy.rs

extern crate allocator_dummy;

fn main() {
    unsafe {
        let before = allocator_dummy::HITS;
        let b = Box::new(3);
        assert_eq!(allocator_dummy::HITS - before, 1);
        drop(b);
        assert_eq!(allocator_dummy::HITS - before, 2);
    }
}
