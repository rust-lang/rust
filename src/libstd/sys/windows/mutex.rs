// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::v1::*;

use sync::atomic::{AtomicUint, ATOMIC_UINT_INIT, Ordering};
use alloc::{self, heap};

use libc::DWORD;
use sys::sync as ffi;

const SPIN_COUNT: DWORD = 4000;

pub struct Mutex { inner: AtomicUint }

pub const MUTEX_INIT: Mutex = Mutex { inner: ATOMIC_UINT_INIT };

unsafe impl Sync for Mutex {}

#[inline]
pub unsafe fn raw(m: &Mutex) -> ffi::LPCRITICAL_SECTION {
    m.get()
}

impl Mutex {
    #[inline]
    pub unsafe fn new() -> Mutex {
        Mutex { inner: AtomicUint::new(init_lock() as uint) }
    }
    #[inline]
    pub unsafe fn lock(&self) {
        ffi::EnterCriticalSection(self.get())
    }
    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        ffi::TryEnterCriticalSection(self.get()) != 0
    }
    #[inline]
    pub unsafe fn unlock(&self) {
        ffi::LeaveCriticalSection(self.get())
    }
    pub unsafe fn destroy(&self) {
        let lock = self.inner.swap(0, Ordering::SeqCst);
        if lock != 0 { free_lock(lock as ffi::LPCRITICAL_SECTION) }
    }

    unsafe fn get(&self) -> ffi::LPCRITICAL_SECTION {
        match self.inner.load(Ordering::SeqCst) {
            0 => {}
            n => return n as ffi::LPCRITICAL_SECTION
        }
        let lock = init_lock();
        match self.inner.compare_and_swap(0, lock as uint, Ordering::SeqCst) {
            0 => return lock as ffi::LPCRITICAL_SECTION,
            _ => {}
        }
        free_lock(lock);
        return self.inner.load(Ordering::SeqCst) as ffi::LPCRITICAL_SECTION;
    }
}

unsafe fn init_lock() -> ffi::LPCRITICAL_SECTION {
    let block = heap::allocate(ffi::CRITICAL_SECTION_SIZE, 8)
                        as ffi::LPCRITICAL_SECTION;
    if block.is_null() { alloc::oom() }
    ffi::InitializeCriticalSectionAndSpinCount(block, SPIN_COUNT);
    return block;
}

unsafe fn free_lock(h: ffi::LPCRITICAL_SECTION) {
    ffi::DeleteCriticalSection(h);
    heap::deallocate(h as *mut _, ffi::CRITICAL_SECTION_SIZE, 8);
}
