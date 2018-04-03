// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use alloc::{Excess, Layout, AllocErr, CannotReallocInPlace};
use core::alloc::Alloc as CoreAlloc;

#[doc(hidden)]
pub mod __core {
    pub use core::*;
}

/// Compatibility with older versions of #[global_allocator] during bootstrap
pub unsafe trait Alloc {
    unsafe fn alloc(&mut self, layout: Layout) -> Result<*mut u8, AllocErr>;
    unsafe fn dealloc(&mut self, ptr: *mut u8, layout: Layout);
    fn oom(&mut self, err: AllocErr) -> !;
    fn usable_size(&self, layout: &Layout) -> (usize, usize);
    unsafe fn realloc(&mut self,
                      ptr: *mut u8,
                      layout: Layout,
                      new_layout: Layout) -> Result<*mut u8, AllocErr>;
    unsafe fn alloc_zeroed(&mut self, layout: Layout) -> Result<*mut u8, AllocErr>;
    unsafe fn alloc_excess(&mut self, layout: Layout) -> Result<Excess, AllocErr>;
    unsafe fn realloc_excess(&mut self,
                             ptr: *mut u8,
                             layout: Layout,
                             new_layout: Layout) -> Result<Excess, AllocErr>;
    unsafe fn grow_in_place(&mut self,
                            ptr: *mut u8,
                            layout: Layout,
                            new_layout: Layout) -> Result<(), CannotReallocInPlace>;
    unsafe fn shrink_in_place(&mut self,
                              ptr: *mut u8,
                              layout: Layout,
                              new_layout: Layout) -> Result<(), CannotReallocInPlace>;
}

#[allow(deprecated)]
unsafe impl<T> Alloc for T where T: CoreAlloc {
    unsafe fn alloc(&mut self, layout: Layout) -> Result<*mut u8, AllocErr> {
        CoreAlloc::alloc(self, layout)
    }

    unsafe fn dealloc(&mut self, ptr: *mut u8, layout: Layout) {
        CoreAlloc::dealloc(self, ptr, layout)
    }

    fn oom(&mut self, err: AllocErr) -> ! {
        CoreAlloc::oom(self, err)
    }

    fn usable_size(&self, layout: &Layout) -> (usize, usize) {
        CoreAlloc::usable_size(self, layout)
    }

    unsafe fn realloc(&mut self,
                      ptr: *mut u8,
                      layout: Layout,
                      new_layout: Layout) -> Result<*mut u8, AllocErr> {
        CoreAlloc::realloc(self, ptr, layout, new_layout)
    }

    unsafe fn alloc_zeroed(&mut self, layout: Layout) -> Result<*mut u8, AllocErr> {
        CoreAlloc::alloc_zeroed(self, layout)
    }

    unsafe fn alloc_excess(&mut self, layout: Layout) -> Result<Excess, AllocErr> {
        CoreAlloc::alloc_excess(self, layout)
    }

    unsafe fn realloc_excess(&mut self,
                             ptr: *mut u8,
                             layout: Layout,
                             new_layout: Layout) -> Result<Excess, AllocErr> {
        CoreAlloc::realloc_excess(self, ptr, layout, new_layout)
    }

    unsafe fn grow_in_place(&mut self,
                            ptr: *mut u8,
                            layout: Layout,
                            new_layout: Layout) -> Result<(), CannotReallocInPlace> {
        CoreAlloc::grow_in_place(self, ptr, layout, new_layout)
    }

    unsafe fn shrink_in_place(&mut self,
                              ptr: *mut u8,
                              layout: Layout,
                              new_layout: Layout) -> Result<(), CannotReallocInPlace> {
        CoreAlloc::shrink_in_place(self, ptr, layout, new_layout)
    }
}
