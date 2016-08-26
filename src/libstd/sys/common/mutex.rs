// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use sys::mutex as imp;

/// An OS-based mutual exclusion lock.
///
/// This is the thinnest cross-platform wrapper around OS mutexes. All usage of
/// this mutex is unsafe and it is recommended to instead use the safe wrapper
/// at the top level of the crate instead of this type.
pub struct Mutex(imp::Mutex);

unsafe impl Sync for Mutex {}

impl Mutex {
    /// Creates a new mutex for use.
    ///
    /// Behavior is undefined if the mutex is moved after it is
    /// first used with any of the functions below.
    pub const fn new() -> Mutex { Mutex(imp::Mutex::new()) }

    /// Prepare the mutex for use.
    ///
    /// This should be called once the mutex is at a stable memory address.
    #[inline]
    pub unsafe fn init(&mut self) { self.0.init() }

    /// Locks the mutex blocking the current thread until it is available.
    ///
    /// Behavior is undefined if the mutex has been moved between this and any
    /// previous function call.
    #[inline]
    pub unsafe fn lock(&self) { self.0.lock() }

    /// Attempts to lock the mutex without blocking, returning whether it was
    /// successfully acquired or not.
    ///
    /// Behavior is undefined if the mutex has been moved between this and any
    /// previous function call.
    #[inline]
    pub unsafe fn try_lock(&self) -> bool { self.0.try_lock() }

    /// Unlocks the mutex.
    ///
    /// Behavior is undefined if the current thread does not actually hold the
    /// mutex.
    #[inline]
    pub unsafe fn unlock(&self) { self.0.unlock() }

    /// Deallocates all resources associated with this mutex.
    ///
    /// Behavior is undefined if there are current or will be future users of
    /// this mutex.
    #[inline]
    pub unsafe fn destroy(&self) { self.0.destroy() }
}

// not meant to be exported to the outside world, just the containing module
pub fn raw(mutex: &Mutex) -> &imp::Mutex { &mutex.0 }
