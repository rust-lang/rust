// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use sys::mutex::raw;

use sys::mutex as imp;

/// An OS-based mutual exclusion lock.
///
/// This is the thinnest cross-platform wrapper around OS mutexes. All usage of
/// this mutex is unsafe and it is recommended to instead use the safe wrapper
/// at the top level of the crate instead of this type.
pub struct Mutex(imp::Mutex);

/// Constant initializer for statically allocated mutexes.
pub const MUTEX_INIT: Mutex = Mutex(imp::MUTEX_INIT);

impl Mutex {
    /// Creates a newly initialized mutex.
    ///
    /// Behavior is undefined if the mutex is moved after the first method is
    /// called on the mutex.
    #[inline]
    pub unsafe fn new() -> Mutex { Mutex(imp::Mutex::new()) }

    /// Lock the mutex blocking the current thread until it is available.
    ///
    /// Behavior is undefined if the mutex has been moved between this and any
    /// previous function call.
    #[inline]
    pub unsafe fn lock(&self) { self.0.lock() }

    /// Attempt to lock the mutex without blocking, returning whether it was
    /// successfully acquired or not.
    ///
    /// Behavior is undefined if the mutex has been moved between this and any
    /// previous function call.
    #[inline]
    pub unsafe fn try_lock(&self) -> bool { self.0.try_lock() }

    /// Unlock the mutex.
    ///
    /// Behavior is undefined if the current thread does not actually hold the
    /// mutex.
    #[inline]
    pub unsafe fn unlock(&self) { self.0.unlock() }

    /// Deallocate all resources associated with this mutex.
    ///
    /// Behavior is undefined if there are current or will be future users of
    /// this mutex.
    #[inline]
    pub unsafe fn destroy(&self) { self.0.destroy() }
}

// not meant to be exported to the outside world, just the containing module
pub fn raw(mutex: &Mutex) -> &imp::Mutex { &mutex.0 }
