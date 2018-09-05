// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use num::NonZeroUsize;

use super::waitqueue::{WaitVariable, WaitQueue, SpinMutex, NotifiedTcs, try_lock_or_false};

pub struct RWLock {
    readers: SpinMutex<WaitVariable<Option<NonZeroUsize>>>,
    writer: SpinMutex<WaitVariable<bool>>,
}

//unsafe impl Send for RWLock {}
//unsafe impl Sync for RWLock {} // FIXME

impl RWLock {
    #[unstable(feature = "sgx_internals", issue = "0")] // FIXME: min_const_fn
    pub const fn new() -> RWLock {
        RWLock {
            readers: SpinMutex::new(WaitVariable::new(None)),
            writer: SpinMutex::new(WaitVariable::new(false))
        }
    }

    #[inline]
    pub unsafe fn read(&self) {
        let mut rguard = self.readers.lock();
        let wguard = self.writer.lock();
        if *wguard.lock_var() || !wguard.queue_empty() {
            // Another thread has or is waiting for the write lock, wait
            drop(wguard);
            WaitQueue::wait(rguard);
            // Another thread has passed the lock to us
        } else {
            // No waiting writers, acquire the read lock
            *rguard.lock_var_mut() =
                NonZeroUsize::new(rguard.lock_var().map_or(0, |n| n.get()) + 1);
        }
    }

    #[inline]
    pub unsafe fn try_read(&self) -> bool {
        let mut rguard = try_lock_or_false!(self.readers);
        let wguard = try_lock_or_false!(self.writer);
        if *wguard.lock_var() || !wguard.queue_empty() {
            // Another thread has or is waiting for the write lock
            false
        } else {
            // No waiting writers, acquire the read lock
            *rguard.lock_var_mut() =
                NonZeroUsize::new(rguard.lock_var().map_or(0, |n| n.get()) + 1);
            true
        }
    }

    #[inline]
    pub unsafe fn write(&self) {
        let rguard = self.readers.lock();
        let mut wguard = self.writer.lock();
        if *wguard.lock_var() || rguard.lock_var().is_some() {
            // Another thread has the lock, wait
            drop(rguard);
            WaitQueue::wait(wguard);
            // Another thread has passed the lock to us
        } else {
            // We are just now obtaining the lock
            *wguard.lock_var_mut() = true;
        }
    }

    #[inline]
    pub unsafe fn try_write(&self) -> bool {
        let rguard = try_lock_or_false!(self.readers);
        let mut wguard = try_lock_or_false!(self.writer);
        if *wguard.lock_var() || rguard.lock_var().is_some() {
            // Another thread has the lock
            false
        } else {
            // We are just now obtaining the lock
            *wguard.lock_var_mut() = true;
            true
        }
    }

    #[inline]
    pub unsafe fn read_unlock(&self) {
        let mut rguard = self.readers.lock();
        let wguard = self.writer.lock();
        *rguard.lock_var_mut() = NonZeroUsize::new(rguard.lock_var().unwrap().get() - 1);
        if rguard.lock_var().is_some() {
            // There are other active readers
        } else {
            if let Ok(mut wguard) = WaitQueue::notify_one(wguard) {
                // A writer was waiting, pass the lock
                *wguard.lock_var_mut() = true;
            } else {
                // No writers were waiting, the lock is released
                assert!(rguard.queue_empty());
            }
        }
    }

    #[inline]
    pub unsafe fn write_unlock(&self) {
        let rguard = self.readers.lock();
        let wguard = self.writer.lock();
        if let Err(mut wguard) = WaitQueue::notify_one(wguard) {
            // No writers waiting, release the write lock
            *wguard.lock_var_mut() = false;
            if let Ok(mut rguard) = WaitQueue::notify_all(rguard) {
                // One or more readers were waiting, pass the lock to them
                if let NotifiedTcs::All { count } = rguard.notified_tcs() {
                    *rguard.lock_var_mut() = Some(count)
                } else {
                    unreachable!() // called notify_all
                }
            } else {
                // No readers waiting, the lock is released
            }
        } else {
            // There was a thread waiting for write, just pass the lock
        }
    }

    #[inline]
    pub unsafe fn destroy(&self) {}
}
