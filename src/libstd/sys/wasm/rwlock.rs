// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use cell::UnsafeCell;

pub struct RWLock {
    mode: UnsafeCell<isize>,
}

unsafe impl Send for RWLock {}
unsafe impl Sync for RWLock {} // no threads on wasm

impl RWLock {
    pub const fn new() -> RWLock {
        RWLock {
            mode: UnsafeCell::new(0),
        }
    }

    #[inline]
    pub unsafe fn read(&self) {
        let mode = self.mode.get();
        if *mode >= 0 {
            *mode += 1;
        } else {
            panic!("rwlock locked for writing");
        }
    }

    #[inline]
    pub unsafe fn try_read(&self) -> bool {
        let mode = self.mode.get();
        if *mode >= 0 {
            *mode += 1;
            true
        } else {
            false
        }
    }

    #[inline]
    pub unsafe fn write(&self) {
        let mode = self.mode.get();
        if *mode == 0 {
            *mode = -1;
        } else {
            panic!("rwlock locked for reading")
        }
    }

    #[inline]
    pub unsafe fn try_write(&self) -> bool {
        let mode = self.mode.get();
        if *mode == 0 {
            *mode = -1;
            true
        } else {
            false
        }
    }

    #[inline]
    pub unsafe fn read_unlock(&self) {
        *self.mode.get() -= 1;
    }

    #[inline]
    pub unsafe fn write_unlock(&self) {
        *self.mode.get() += 1;
    }

    #[inline]
    pub unsafe fn destroy(&self) {
    }
}
