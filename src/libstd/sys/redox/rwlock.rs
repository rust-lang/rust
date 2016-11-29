// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::mutex::Mutex;

pub struct RWLock {
    mutex: Mutex
}

unsafe impl Send for RWLock {}
unsafe impl Sync for RWLock {}

impl RWLock {
    pub const fn new() -> RWLock {
        RWLock {
            mutex: Mutex::new()
        }
    }

    #[inline]
    pub unsafe fn read(&self) {
        self.mutex.lock();
    }

    #[inline]
    pub unsafe fn try_read(&self) -> bool {
        self.mutex.try_lock()
    }

    #[inline]
    pub unsafe fn write(&self) {
        self.mutex.lock();
    }

    #[inline]
    pub unsafe fn try_write(&self) -> bool {
        self.mutex.try_lock()
    }

    #[inline]
    pub unsafe fn read_unlock(&self) {
        self.mutex.unlock();
    }

    #[inline]
    pub unsafe fn write_unlock(&self) {
        self.mutex.unlock();
    }

    #[inline]
    pub unsafe fn destroy(&self) {
        self.mutex.destroy();
    }
}
