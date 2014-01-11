// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(missing_doc)];
use ops::Drop;
use libc::{malloc, free, size_t, c_void};
use gc;
use gc::Trace;
use mem;
use ptr;
use ptr::RawPtr;
use unstable::intrinsics::move_val_init;

fn trace<T: Trace>(ptr: *(), _: uint, tracer: &mut gc::GcTracer) {
    unsafe {
        (*(ptr as *T)).trace(tracer)
    }
}

#[unsafe_no_drop_flag]
pub struct Uniq<T> {
    priv ptr: *mut T
}

impl<T: Trace> Uniq<T> {
    pub fn new(value: T) -> Uniq<T> {
        unsafe {
            let ptr = malloc(mem::size_of::<T>() as size_t) as *mut T;
            gc::register_root_changes([], [(ptr as *T, 0, trace::<T>)]);
            move_val_init(&mut *ptr, value);
            Uniq { ptr: ptr }
        }
    }

    pub fn borrow<'a>(&'a self) -> &'a T {
        unsafe { &*self.ptr }
    }
    pub fn borrow_mut<'a>(&'a mut self) -> &'a mut T {
        unsafe { &mut *self.ptr }
    }
    pub fn move(mut self) -> T {
        unsafe {
            let val = ptr::read_ptr(self.ptr as *T);
            drop_no_inner_dtor(&mut self);
            val
        }
    }
}

unsafe fn drop_no_inner_dtor<T>(x: &mut Uniq<T>) {
    gc::register_root_changes([x.ptr as *T], []);
    free(x.ptr as *c_void);
    x.ptr = 0 as *mut T;
}

#[unsafe_destructor]
impl<T> Drop for Uniq<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ptr::read_ptr(self.ptr as *T);
                drop_no_inner_dtor(self)
            }
        }
    }
}

impl<T: Trace> Trace for Uniq<T> {
    fn trace(&self, tracer: &mut gc::GcTracer) {
        if tracer.pointer_first_trace(self.ptr as *()) {
            unsafe {
                (*self.ptr).trace(tracer)
            }
        }
    }
}
