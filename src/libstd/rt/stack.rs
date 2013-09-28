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
use ptr::RawPtr;
use vec;
use ops::Drop;
use libc::{c_uint, uintptr_t};

pub struct StackSegment {
    buf: ~[u8],
    valgrind_id: c_uint
}

impl StackSegment {
    pub fn new(size: uint) -> StackSegment {
        #[fixed_stack_segment]; #[inline(never)];

        unsafe {
            // Crate a block of uninitialized values
            let mut stack = vec::with_capacity(size);
            vec::raw::set_len(&mut stack, size);

            let mut stk = StackSegment {
                buf: stack,
                valgrind_id: 0
            };

            // XXX: Using the FFI to call a C macro. Slow
            stk.valgrind_id = rust_valgrind_stack_register(stk.start(), stk.end());
            return stk;
        }
    }

    /// Point to the low end of the allocated stack
    pub fn start(&self) -> *uint {
        vec::raw::to_ptr(self.buf) as *uint
    }

    /// Point one word beyond the high end of the allocated stack
    pub fn end(&self) -> *uint {
        unsafe {
            vec::raw::to_ptr(self.buf).offset(self.buf.len() as int) as *uint
        }
    }
}

impl Drop for StackSegment {
    fn drop(&mut self) {
        #[fixed_stack_segment]; #[inline(never)];

        unsafe {
            // XXX: Using the FFI to call a C macro. Slow
            rust_valgrind_stack_deregister(self.valgrind_id);
        }
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

extern {
    fn rust_valgrind_stack_register(start: *uintptr_t, end: *uintptr_t) -> c_uint;
    fn rust_valgrind_stack_deregister(id: c_uint);
}
