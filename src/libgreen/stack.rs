// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::rt::env::max_cached_stacks;
use std::os::{errno, page_size, MemoryMap, MapReadable, MapWritable, MapNonStandardFlags};
#[cfg(not(windows))]
use std::libc::{MAP_STACK, MAP_PRIVATE, MAP_ANON};
use std::libc::{c_uint, c_int, c_void, uintptr_t};

/// A task's stack. The name "Stack" is a vestige of segmented stacks.
pub struct Stack {
    priv buf: MemoryMap,
    priv min_size: uint,
    priv valgrind_id: c_uint,
}

// This is what glibc uses on linux. MAP_STACK can be 0 if any platform we decide to support doesn't
// have it defined. Considering also using MAP_GROWSDOWN
#[cfg(not(windows))]
static STACK_FLAGS: c_int = MAP_STACK | MAP_PRIVATE | MAP_ANON;
#[cfg(windows)]
static STACK_FLAGS: c_int = 0;

impl Stack {
    pub fn new(size: uint) -> Stack {
        // Map in a stack. Eventually we might be able to handle stack allocation failure, which
        // would fail to spawn the task. But there's not many sensible things to do on OOM.
        // Failure seems fine (and is what the old stack allocation did).
        let stack = MemoryMap::new(size, [MapReadable, MapWritable,
                                          MapNonStandardFlags(STACK_FLAGS)]).unwrap();

        // Change the last page to be inaccessible. This is to provide safety; when an FFI
        // function overflows it will (hopefully) hit this guard page. It isn't guaranteed, but
        // that's why FFI is unsafe. buf.data is guaranteed to be aligned properly.
        if !protect_last_page(&stack) {
            fail!("Could not memory-protect guard page. stack={:?}, errno={}",
                  stack, errno());
        }

        let mut stk = Stack {
            buf: stack,
            min_size: size,
            valgrind_id: 0
        };

        // XXX: Using the FFI to call a C macro. Slow
        stk.valgrind_id = unsafe { rust_valgrind_stack_register(stk.start(), stk.end()) };
        return stk;
    }

    /// Point to the low end of the allocated stack
    pub fn start(&self) -> *uint {
        self.buf.data as *uint
    }

    /// Point one word beyond the high end of the allocated stack
    pub fn end(&self) -> *uint {
        unsafe {
            self.buf.data.offset(self.buf.len as int) as *uint
        }
    }
}

// These use ToPrimitive so that we never need to worry about the sizes of whatever types these
// (which we would with scalar casts). It's either a wrapper for a scalar cast or failure: fast, or
// will fail during compilation.
#[cfg(unix)]
fn protect_last_page(stack: &MemoryMap) -> bool {
    use std::libc::{mprotect, PROT_NONE, size_t};
    unsafe {
        // This may seem backwards: the start of the segment is the last page? Yes! The stack grows
        // from higher addresses (the end of the allocated block) to lower addresses (the start of
        // the allocated block).
        let last_page = stack.data as *c_void;
        mprotect(last_page, page_size() as size_t, PROT_NONE) != -1
    }
}

#[cfg(windows)]
fn protect_last_page(stack: &MemoryMap) -> bool {
    use std::libc::{VirtualProtect, PAGE_NOACCESS, SIZE_T, LPDWORD, DWORD};
    unsafe {
        // see above
        let last_page = stack.data as *mut c_void;
        let old_prot: DWORD = 0;
        VirtualProtect(last_page, page_size() as SIZE_T, PAGE_NOACCESS, &mut old_prot as *mut LPDWORD) != 0
    }
}

impl Drop for Stack {
    fn drop(&mut self) {
        unsafe {
            // XXX: Using the FFI to call a C macro. Slow
            rust_valgrind_stack_deregister(self.valgrind_id);
        }
    }
}

pub struct StackPool {
    // Ideally this would be some datastructure that preserved ordering on Stack.min_size.
    priv stacks: ~[Stack],
}

impl StackPool {
    pub fn new() -> StackPool {
        StackPool {
            stacks: ~[],
        }
    }

    pub fn take_stack(&mut self, min_size: uint) -> Stack {
        // Ideally this would be a binary search
        match self.stacks.iter().position(|s| s.min_size < min_size) {
            Some(idx) => self.stacks.swap_remove(idx),
            None      => Stack::new(min_size)
        }
    }

    pub fn give_stack(&mut self, stack: Stack) {
        if self.stacks.len() <= max_cached_stacks() {
            self.stacks.push(stack)
        }
    }
}

extern {
    fn rust_valgrind_stack_register(start: *uintptr_t, end: *uintptr_t) -> c_uint;
    fn rust_valgrind_stack_deregister(id: c_uint);
}
