// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// Simple refcount structure for cloning handles
///
/// This is meant to be an unintrusive solution to cloning handles in rustuv.
/// The handles themselves shouldn't be sharing memory because there are bits of
/// state in the rust objects which shouldn't be shared across multiple users of
/// the same underlying uv object, hence Rc is not used and this simple counter
/// should suffice.

use alloc::arc::Arc;
use std::cell::UnsafeCell;

pub struct Refcount {
    rc: Arc<UnsafeCell<uint>>,
}

impl Refcount {
    /// Creates a new refcount of 1
    pub fn new() -> Refcount {
        Refcount { rc: Arc::new(UnsafeCell::new(1)) }
    }

    fn increment(&self) {
        unsafe { *self.rc.get() += 1; }
    }

    /// Returns whether the refcount just hit 0 or not
    pub fn decrement(&self) -> bool {
        unsafe {
            *self.rc.get() -= 1;
            *self.rc.get() == 0
        }
    }
}

impl Clone for Refcount {
    fn clone(&self) -> Refcount {
        self.increment();
        Refcount { rc: self.rc.clone() }
    }
}
