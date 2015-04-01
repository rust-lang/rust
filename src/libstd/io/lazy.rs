// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::v1::*;

use boxed;
use cell::UnsafeCell;
use rt;
use sync::{StaticMutex, Arc};

pub struct Lazy<T: Send + Sync> {
    pub lock: StaticMutex,
    pub ptr: UnsafeCell<*mut Arc<T>>,
    pub init: fn() -> Arc<T>,
}

unsafe impl<T: Send + Sync> Sync for Lazy<T> {}

macro_rules! lazy_init {
    ($init:expr) => (::io::lazy::Lazy {
        lock: ::sync::MUTEX_INIT,
        ptr: ::cell::UnsafeCell { value: 0 as *mut _ },
        init: $init,
    })
}

impl<T: Send + Sync + 'static> Lazy<T> {
    pub fn get(&'static self) -> Option<Arc<T>> {
        let _g = self.lock.lock();
        unsafe {
            let ptr = *self.ptr.get();
            if ptr.is_null() {
                Some(self.init())
            } else if ptr as usize == 1 {
                None
            } else {
                Some((*ptr).clone())
            }
        }
    }

    unsafe fn init(&'static self) -> Arc<T> {
        // If we successfully register an at exit handler, then we cache the
        // `Arc` allocation in our own internal box (it will get deallocated by
        // the at exit handler). Otherwise we just return the freshly allocated
        // `Arc`.
        let registered = rt::at_exit(move || {
            let g = self.lock.lock();
            let ptr = *self.ptr.get();
            *self.ptr.get() = 1 as *mut _;
            drop(g);
            drop(Box::from_raw(ptr))
        });
        let ret = (self.init)();
        if registered.is_ok() {
            *self.ptr.get() = boxed::into_raw(Box::new(ret.clone()));
        }
        return ret
    }
}
