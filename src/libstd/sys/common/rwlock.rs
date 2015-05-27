// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use sys::rwlock as imp;

/// An OS-based reader-writer lock.
///
/// This structure is entirely unsafe and serves as the lowest layer of a
/// cross-platform binding of system rwlocks. It is recommended to use the
/// safer types at the top level of this crate instead of this type.
pub struct RWLock(imp::RWLock);

impl RWLock {
    /// Creates a new reader-writer lock for use.
    ///
    /// Behavior is undefined if the reader-writer lock is moved after it is
    /// first used with any of the functions below.
    pub const fn new() -> RWLock { RWLock(imp::RWLock::new()) }

    /// Acquires shared access to the underlying lock, blocking the current
    /// thread to do so.
    ///
    /// Behavior is undefined if the rwlock has been moved between this and any
    /// previous method call.
    #[inline]
    pub unsafe fn read(&self) { self.0.read() }

    /// Attempts to acquire shared access to this lock, returning whether it
    /// succeeded or not.
    ///
    /// This function does not block the current thread.
    ///
    /// Behavior is undefined if the rwlock has been moved between this and any
    /// previous method call.
    #[inline]
    pub unsafe fn try_read(&self) -> bool { self.0.try_read() }

    /// Acquires write access to the underlying lock, blocking the current thread
    /// to do so.
    ///
    /// Behavior is undefined if the rwlock has been moved between this and any
    /// previous method call.
    #[inline]
    pub unsafe fn write(&self) { self.0.write() }

    /// Attempts to acquire exclusive access to this lock, returning whether it
    /// succeeded or not.
    ///
    /// This function does not block the current thread.
    ///
    /// Behavior is undefined if the rwlock has been moved between this and any
    /// previous method call.
    #[inline]
    pub unsafe fn try_write(&self) -> bool { self.0.try_write() }

    /// Unlocks previously acquired shared access to this lock.
    ///
    /// Behavior is undefined if the current thread does not have shared access.
    #[inline]
    pub unsafe fn read_unlock(&self) { self.0.read_unlock() }

    /// Unlocks previously acquired exclusive access to this lock.
    ///
    /// Behavior is undefined if the current thread does not currently have
    /// exclusive access.
    #[inline]
    pub unsafe fn write_unlock(&self) { self.0.write_unlock() }

    /// Destroys OS-related resources with this RWLock.
    ///
    /// Behavior is undefined if there are any currently active users of this
    /// lock.
    #[inline]
    pub unsafe fn destroy(&self) { self.0.destroy() }
}
