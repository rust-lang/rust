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
use std::os::{errno, page_size, MemoryMap, MapReadable, MapWritable,
              MapNonStandardFlags, MapVirtual};
use std::libc;

/// A task's stack. The name "Stack" is a vestige of segmented stacks.
pub struct Stack {
    priv buf: MemoryMap,
    priv min_size: uint,
    priv valgrind_id: libc::c_uint,
}

// Try to use MAP_STACK on platforms that support it (it's what we're doing
// anyway), but some platforms don't support it at all. For example, it appears
// that there's a bug in freebsd that MAP_STACK implies MAP_FIXED (so it always
// fails): http://lists.freebsd.org/pipermail/freebsd-bugs/2011-July/044840.html
#[cfg(not(windows), not(target_os = "freebsd"))]
static STACK_FLAGS: libc::c_int = libc::MAP_STACK | libc::MAP_PRIVATE |
                                  libc::MAP_ANON;
#[cfg(target_os = "freebsd")]
static STACK_FLAGS: libc::c_int = libc::MAP_PRIVATE | libc::MAP_ANON;
#[cfg(windows)]
static STACK_FLAGS: libc::c_int = 0;

impl Stack {
    /// Allocate a new stack of `size`. If size = 0, this will fail. Use
    /// `dummy_stack` if you want a zero-sized stack.
    pub fn new(size: uint) -> Stack {
        // Map in a stack. Eventually we might be able to handle stack
        // allocation failure, which would fail to spawn the task. But there's
        // not many sensible things to do on OOM.  Failure seems fine (and is
        // what the old stack allocation did).
        let stack = match MemoryMap::new(size, [MapReadable, MapWritable,
                                         MapNonStandardFlags(STACK_FLAGS)]) {
            Ok(map) => map,
            Err(e) => fail!("mmap for stack of size {} failed: {}", size, e)
        };

        // Change the last page to be inaccessible. This is to provide safety;
        // when an FFI function overflows it will (hopefully) hit this guard
        // page. It isn't guaranteed, but that's why FFI is unsafe. buf.data is
        // guaranteed to be aligned properly.
        if !protect_last_page(&stack) {
            fail!("Could not memory-protect guard page. stack={:?}, errno={}",
                  stack, errno());
        }

        let mut stk = Stack {
            buf: stack,
            min_size: size,
            valgrind_id: 0
        };

        // FIXME: Using the FFI to call a C macro. Slow
        stk.valgrind_id = unsafe {
            rust_valgrind_stack_register(stk.start(), stk.end())
        };
        return stk;
    }

    /// Create a 0-length stack which starts (and ends) at 0.
    pub unsafe fn dummy_stack() -> Stack {
        Stack {
            buf: MemoryMap { data: 0 as *mut u8, len: 0, kind: MapVirtual },
            min_size: 0,
            valgrind_id: 0
        }
    }

    /// Point to the low end of the allocated stack
    pub fn start(&self) -> *uint {
        self.buf.data as *uint
    }

    /// Point one uint beyond the high end of the allocated stack
    pub fn end(&self) -> *uint {
        unsafe {
            self.buf.data.offset(self.buf.len as int) as *uint
        }
    }
}

#[cfg(unix)]
fn protect_last_page(stack: &MemoryMap) -> bool {
    unsafe {
        // This may seem backwards: the start of the segment is the last page?
        // Yes! The stack grows from higher addresses (the end of the allocated
        // block) to lower addresses (the start of the allocated block).
        let last_page = stack.data as *libc::c_void;
        libc::mprotect(last_page, page_size() as libc::size_t,
                       libc::PROT_NONE) != -1
    }
}

#[cfg(windows)]
fn protect_last_page(stack: &MemoryMap) -> bool {
    unsafe {
        // see above
        let last_page = stack.data as *mut libc::c_void;
        let mut old_prot: libc::DWORD = 0;
        libc::VirtualProtect(last_page, page_size() as libc::SIZE_T,
                             libc::PAGE_NOACCESS,
                             &mut old_prot as libc::LPDWORD) != 0
    }
}

impl Drop for Stack {
    fn drop(&mut self) {
        unsafe {
            // FIXME: Using the FFI to call a C macro. Slow
            rust_valgrind_stack_deregister(self.valgrind_id);
        }
    }
}

pub struct StackPool {
    // Ideally this would be some datastructure that preserved ordering on
    // Stack.min_size.
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
        match self.stacks.iter().position(|s| min_size <= s.min_size) {
            Some(idx) => self.stacks.swap_remove(idx).unwrap(),
            None => Stack::new(min_size)
        }
    }

    pub fn give_stack(&mut self, stack: Stack) {
        if self.stacks.len() <= max_cached_stacks() {
            self.stacks.push(stack)
        }
    }
}

extern {
    fn rust_valgrind_stack_register(start: *libc::uintptr_t,
                                    end: *libc::uintptr_t) -> libc::c_uint;
    fn rust_valgrind_stack_deregister(id: libc::c_uint);
}

#[cfg(test)]
mod tests {
    use super::StackPool;

    #[test]
    fn stack_pool_caches() {
        let mut p = StackPool::new();
        let s = p.take_stack(10);
        p.give_stack(s);
        let s = p.take_stack(4);
        assert_eq!(s.min_size, 10);
        p.give_stack(s);
        let s = p.take_stack(14);
        assert_eq!(s.min_size, 14);
        p.give_stack(s);
    }

    #[test]
    fn stack_pool_caches_exact() {
        let mut p = StackPool::new();
        let mut s = p.take_stack(10);
        s.valgrind_id = 100;
        p.give_stack(s);

        let s = p.take_stack(10);
        assert_eq!(s.min_size, 10);
        assert_eq!(s.valgrind_id, 100);
    }
}
