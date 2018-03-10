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

pub struct Mutex {
    locked: UnsafeCell<bool>,
}

unsafe impl Send for Mutex {}
unsafe impl Sync for Mutex {} // no threads on wasm

impl Mutex {
    pub const fn new() -> Mutex {
        Mutex { locked: UnsafeCell::new(false) }
    }

    #[inline]
    pub unsafe fn init(&mut self) {
    }

    #[inline]
    pub unsafe fn lock(&self) {
        let locked = self.locked.get();
        assert!(!*locked, "cannot recursively acquire mutex");
        *locked = true;
    }

    #[inline]
    pub unsafe fn unlock(&self) {
        *self.locked.get() = false;
    }

    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        let locked = self.locked.get();
        if *locked {
            false
        } else {
            *locked = true;
            true
        }
    }

    #[inline]
    pub unsafe fn destroy(&self) {
    }
}

// All empty stubs because wasm has no threads yet, so lock acquisition always
// succeeds.
pub struct ReentrantMutex {
}

impl ReentrantMutex {
    pub unsafe fn uninitialized() -> ReentrantMutex {
        ReentrantMutex { }
    }

    pub unsafe fn init(&mut self) {}

    pub unsafe fn lock(&self) {}

    #[inline]
    pub unsafe fn try_lock(&self) -> bool {
        true
    }

    pub unsafe fn unlock(&self) {}

    pub unsafe fn destroy(&self) {}
}
