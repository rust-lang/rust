// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use container::Container;
use ptr::Ptr;
use vec;

pub struct StackSegment {
    buf: ~[u8]
}

pub impl StackSegment {
    fn new(size: uint) -> StackSegment {
        // Crate a block of uninitialized values
        let mut stack = vec::with_capacity(size);
        unsafe {
            vec::raw::set_len(&mut stack, size);
        }

        StackSegment {
            buf: stack
        }
    }

    /// Point one word beyond the high end of the allocated stack
    fn end(&self) -> *uint {
        vec::raw::to_ptr(self.buf).offset(self.buf.len()) as *uint
    }
}

pub struct StackPool(());

impl StackPool {
    pub fn new() -> StackPool { StackPool(()) }

    fn take_segment(&self, min_size: uint) -> StackSegment {
        StackSegment::new(min_size)
    }

    fn give_segment(&self, _stack: StackSegment) {
    }
}
