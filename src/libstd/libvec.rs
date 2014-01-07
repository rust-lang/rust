// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[allow(missing_doc)];

use container::Container;
use mem::size_of;
use unstable::intrinsics::move_val_init;
use unstable::raw;
use cast::{forget, transmute};
use libc::{free, malloc, realloc};
use ops::Drop;
use vec::{VecIterator, ImmutableVector};
use libc::{c_void, size_t};
use ptr::{read_ptr, RawPtr};
use num::CheckedMul;
use option::{Option, Some, None};
use iter::{Iterator, DoubleEndedIterator};

pub struct Vec<T> {
    priv len: uint,
    priv cap: uint,
    priv ptr: *mut T
}

impl<T> Vec<T> {
    #[inline(always)]
    pub fn new() -> Vec<T> {
        Vec { len: 0, cap: 0, ptr: 0 as *mut T }
    }

    pub fn with_capacity(capacity: uint) -> Vec<T> {
        if capacity == 0 {
            Vec::new()
        } else {
            let size = capacity.checked_mul(&size_of::<T>()).expect("out of mem");
            let ptr = unsafe { malloc(size as size_t) };
            if ptr.is_null() {
                fail!("null pointer")
            }
            Vec { len: 0, cap: capacity, ptr: ptr as *mut T }
        }
    }
}

impl<T> Container for Vec<T> {
    #[inline(always)]
    fn len(&self) -> uint {
        self.len
    }
}

impl<T> Vec<T> {
    #[inline(always)]
    pub fn capacity(&self) -> uint {
        self.cap
    }

    pub fn reserve(&mut self, capacity: uint) {
        if capacity >= self.len {
            let size = capacity.checked_mul(&size_of::<T>()).expect("out of mem");
            self.cap = capacity;
            unsafe {
                let ptr = realloc(self.ptr as *mut c_void, size as size_t) as *mut T;
                if ptr.is_null() {
                    fail!("null pointer")
                }
                self.ptr = ptr;
            }
        }
    }

    #[inline]
    pub fn shrink_to_fit(&mut self) {
        unsafe {
            if self.len == 0 {
                free(self.ptr as *c_void);
                self.cap = 0;
                self.ptr = 0 as *mut T;
            } else {
                self.ptr = realloc(self.ptr as *mut c_void,
                                   (self.len * size_of::<T>()) as size_t) as *mut T;
                self.cap = self.len;
            }
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            unsafe {
                self.len -= 1;
                Some(read_ptr(self.as_slice().unsafe_ref(self.len())))
            }
        }
    }

    #[inline]
    pub fn push(&mut self, value: T) {
        if self.len == self.cap {
            if self.cap == 0 { self.cap += 2 }
            let old_size = self.cap * size_of::<T>();
            self.cap = self.cap * 2;
            let size = old_size * 2;
            if old_size > size { fail!("out of mem") }
            unsafe {
                self.ptr = realloc(self.ptr as *mut c_void, size as size_t) as *mut T;
            }
        }

        unsafe {
            let end = self.ptr.offset(self.len as int) as *mut T;
            move_val_init(&mut *end, value);
            self.len += 1;
        }
    }

    #[inline]
    pub fn as_slice<'a>(&'a self) -> &'a [T] {
        let slice = raw::Slice { data: self.ptr as *T, len: self.len };
        unsafe { transmute(slice) }
    }

    #[inline]
    pub fn as_mut_slice<'a>(&'a mut self) -> &'a mut [T] {
        let slice = raw::Slice { data: self.ptr as *T, len: self.len };
        unsafe { transmute(slice) }
    }

    pub fn move_iter(self) -> MoveIterator<T> {
        unsafe {
            let iter = transmute(self.as_slice().iter());
            let ptr = self.ptr as *mut u8;
            forget(self);
            MoveIterator { allocation: ptr, iter: iter }
        }
    }
}


#[unsafe_destructor]
impl<T> Drop for Vec<T> {
    fn drop(&mut self) {
        unsafe {
            for x in self.as_slice().iter() {
                read_ptr(x as *T);
            }
            free(self.ptr as *c_void)
        }
    }
}

pub struct MoveIterator<T> {
    priv allocation: *mut u8, // the block of memory allocated for the vector
    priv iter: VecIterator<'static, T>
}

impl<T> Iterator<T> for MoveIterator<T> {
    fn next(&mut self) -> Option<T> {
        unsafe {
            self.iter.next().map(|x| read_ptr(x))
        }
    }

    fn size_hint(&self) -> (uint, Option<uint>) {
        self.iter.size_hint()
    }
}

impl<T> DoubleEndedIterator<T> for MoveIterator<T> {
    fn next_back(&mut self) -> Option<T> {
        unsafe {
            self.iter.next_back().map(|x| read_ptr(x))
        }
    }
}

#[unsafe_destructor]
impl<T> Drop for MoveIterator<T> {
    fn drop(&mut self) {
        // destroy the remaining elements
        for _x in *self {}
        unsafe {
            free(self.allocation as *c_void)
        }
    }
}
