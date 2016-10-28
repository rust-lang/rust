// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub struct RWLock;

unsafe impl Send for RWLock {}
unsafe impl Sync for RWLock {}

impl RWLock {
    pub const fn new() -> RWLock {
        RWLock
    }

    #[inline]
    pub unsafe fn read(&self) {
        unimplemented!();
    }

    #[inline]
    pub unsafe fn try_read(&self) -> bool {
        unimplemented!();
    }

    #[inline]
    pub unsafe fn write(&self) {
        unimplemented!();
    }

    #[inline]
    pub unsafe fn try_write(&self) -> bool {
        unimplemented!();
    }

    #[inline]
    pub unsafe fn read_unlock(&self) {
        unimplemented!();
    }

    #[inline]
    pub unsafe fn write_unlock(&self) {
        unimplemented!();
    }

    #[inline]
    pub unsafe fn destroy(&self) {

    }
}
