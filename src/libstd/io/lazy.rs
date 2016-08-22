// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use cell::Cell;
use ptr;
use sync::Arc;
use sys_common;
use sys_common::mutex::Mutex;

pub struct Lazy<T> {
    lock: Mutex,
    ptr: Cell<*mut Arc<T>>,
    init: fn() -> Arc<T>,
}

unsafe impl<T> Sync for Lazy<T> {}

impl<T: Send + Sync + 'static> Lazy<T> {
    pub const fn new(init: fn() -> Arc<T>) -> Lazy<T> {
        Lazy {
            lock: Mutex::new(),
            ptr: Cell::new(ptr::null_mut()),
            init: init
        }
    }

    pub fn get(&'static self) -> Option<Arc<T>> {
        unsafe {
            self.lock.lock();
            let ptr = self.ptr.get();
            let ret = if ptr.is_null() {
                Some(self.init())
            } else if ptr as usize == 1 {
                None
            } else {
                Some((*ptr).clone())
            };
            self.lock.unlock();
            return ret
        }
    }

    unsafe fn init(&'static self) -> Arc<T> {
        // If we successfully register an at exit handler, then we cache the
        // `Arc` allocation in our own internal box (it will get deallocated by
        // the at exit handler). Otherwise we just return the freshly allocated
        // `Arc`.
        let registered = sys_common::at_exit(move || {
            self.lock.lock();
            let ptr = self.ptr.get();
            self.ptr.set(1 as *mut _);
            self.lock.unlock();
            drop(Box::from_raw(ptr))
        });
        let ret = (self.init)();
        if registered.is_ok() {
            self.ptr.set(Box::into_raw(Box::new(ret.clone())));
        }
        ret
    }
}
