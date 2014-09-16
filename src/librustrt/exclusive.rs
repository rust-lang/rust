// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use core::cell::UnsafeCell;
use mutex;

/// An OS mutex over some data.
///
/// This is not a safe primitive to use, it is unaware of the libgreen
/// scheduler, as well as being easily susceptible to misuse due to the usage of
/// the inner NativeMutex.
///
/// > **Note**: This type is not recommended for general use. The mutex provided
/// >           as part of `libsync` should almost always be favored.
pub struct Exclusive<T> {
    lock: mutex::NativeMutex,
    data: UnsafeCell<T>,
}

/// An RAII guard returned via `lock`
pub struct ExclusiveGuard<'a, T:'a> {
    // FIXME #12808: strange name to try to avoid interfering with
    // field accesses of the contained type via Deref
    _data: &'a mut T,
    _guard: mutex::LockGuard<'a>,
}

impl<T: Send> Exclusive<T> {
    /// Creates a new `Exclusive` which will protect the data provided.
    pub fn new(user_data: T) -> Exclusive<T> {
        Exclusive {
            lock: unsafe { mutex::NativeMutex::new() },
            data: UnsafeCell::new(user_data),
        }
    }

    /// Acquires this lock, returning a guard which the data is accessed through
    /// and from which that lock will be unlocked.
    ///
    /// This method is unsafe due to many of the same reasons that the
    /// NativeMutex itself is unsafe.
    pub unsafe fn lock<'a>(&'a self) -> ExclusiveGuard<'a, T> {
        let guard = self.lock.lock();
        let data = &mut *self.data.get();

        ExclusiveGuard {
            _data: data,
            _guard: guard,
        }
    }
}

impl<'a, T: Send> ExclusiveGuard<'a, T> {
    // The unsafety here should be ok because our loan guarantees that the lock
    // itself is not moving
    pub fn signal(&self) {
        unsafe { self._guard.signal() }
    }
    pub fn wait(&self) {
        unsafe { self._guard.wait() }
    }
}

impl<'a, T: Send> Deref<T> for ExclusiveGuard<'a, T> {
    fn deref<'a>(&'a self) -> &'a T { &*self._data }
}
impl<'a, T: Send> DerefMut<T> for ExclusiveGuard<'a, T> {
    fn deref_mut<'a>(&'a mut self) -> &'a mut T { &mut *self._data }
}

#[cfg(test)]
mod tests {
    use std::prelude::*;
    use alloc::arc::Arc;
    use super::Exclusive;
    use std::task;

    #[test]
    fn exclusive_new_arc() {
        unsafe {
            let mut futures = Vec::new();

            let num_tasks = 10;
            let count = 10;

            let total = Arc::new(Exclusive::new(box 0));

            for _ in range(0u, num_tasks) {
                let total = total.clone();
                let (tx, rx) = channel();
                futures.push(rx);

                task::spawn(proc() {
                    for _ in range(0u, count) {
                        **total.lock() += 1;
                    }
                    tx.send(());
                });
            };

            for f in futures.iter_mut() { f.recv() }

            assert_eq!(**total.lock(), num_tasks * count);
        }
    }
}
