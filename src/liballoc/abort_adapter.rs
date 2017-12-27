// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![unstable(feature = "allocator_api",
            reason = "the precise API and guarantees it provides may be tweaked \
                      slightly, especially to possibly take into account the \
                      types being stored to make room for a future \
                      tracing garbage collector",
            issue = "32838")]

use core::usize;
use core::ptr::NonNull;

use crate::alloc::*;

/// An allocator adapter that blows up by calling `Alloc::oom` on all errors.
///
/// On one hand, concrete allocator implementations should always be written
/// without panicking on user error and OOM to give users maximum
/// flexibility. On the other hand, code that depends on allocation succeeding
/// should depend on `Alloc<Err=!>` to avoid repetitively handling errors from
/// which it cannot recover.
///
/// This adapter bridges the gap, effectively allowing `Alloc<Err=!>` to be
/// implemented by any allocator.
#[derive(Copy, Clone, Debug, Default)]
pub struct AbortAdapter<Alloc>(pub Alloc);

impl<A: AllocHelper> AllocHelper for AbortAdapter<A> {
    type Err = !;
}

unsafe impl<A: Alloc> Alloc for AbortAdapter<A> {
    unsafe fn alloc(&mut self, layout: Layout) -> Result<NonNull<u8>, Self::Err> {
        self.0.alloc(layout).or_else(|_| handle_alloc_error(layout))
    }

    unsafe fn dealloc(&mut self, ptr: NonNull<u8>, layout: Layout) {
        self.0.dealloc(ptr, layout)
    }

    fn usable_size(&self, layout: &Layout) -> (usize, usize) {
        self.0.usable_size(layout)
    }

    unsafe fn realloc(&mut self,
                      ptr: NonNull<u8>,
                      layout: Layout,
                      new_size: usize) -> Result<NonNull<u8>, Self::Err> {
        self.0.realloc(ptr, layout, new_size).or_else(|_| handle_alloc_error(layout))
    }

    unsafe fn alloc_zeroed(&mut self, layout: Layout) -> Result<NonNull<u8>, Self::Err> {
        self.0.alloc_zeroed(layout).or_else(|_| handle_alloc_error(layout))
    }

    unsafe fn alloc_excess(&mut self, layout: Layout) -> Result<Excess, Self::Err> {
        self.0.alloc_excess(layout).or_else(|_| handle_alloc_error(layout))
    }

    unsafe fn grow_in_place(&mut self,
                            ptr: NonNull<u8>,
                            layout: Layout,
                            new_size: usize) -> Result<(), CannotReallocInPlace> {
        self.0.grow_in_place(ptr, layout, new_size)
    }

    unsafe fn shrink_in_place(&mut self,
                              ptr: NonNull<u8>,
                              layout: Layout,
                              new_size: usize) -> Result<(), CannotReallocInPlace> {
        self.0.shrink_in_place(ptr, layout, new_size)
    }

    fn alloc_one<T>(&mut self) -> Result<NonNull<T>, Self::Err>
        where Self: Sized
    {
        self.0.alloc_one().or_else(|_| handle_alloc_error(Layout::new::<T>()))
    }

    unsafe fn dealloc_one<T>(&mut self, ptr: NonNull<T>)
        where Self: Sized
    {
        self.0.dealloc_one(ptr)
    }

    fn alloc_array<T>(&mut self, n: usize) -> Result<NonNull<T>, Self::Err>
        where Self: Sized
    {
        self.0.alloc_array(n).or_else(|_| handle_alloc_error(Layout::new::<T>()))
    }

    unsafe fn realloc_array<T>(&mut self,
                               ptr: NonNull<T>,
                               n_old: usize,
                               n_new: usize) -> Result<NonNull<T>, Self::Err>
        where Self: Sized
    {
        self.0.realloc_array(ptr, n_old, n_new)
            .or_else(|_| handle_alloc_error(Layout::new::<T>()))
    }

    unsafe fn dealloc_array<T>(&mut self, ptr: NonNull<T>, n: usize) -> Result<(), Self::Err>
        where Self: Sized
    {
        self.0.dealloc_array(ptr, n).or_else(|_| handle_alloc_error(Layout::new::<T>()))
    }
}
