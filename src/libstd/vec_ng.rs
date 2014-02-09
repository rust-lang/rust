// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Migrate documentation over from `std::vec` when it is removed.
#[doc(hidden)];

use ops::Drop;
use option::{None, Option, Some};
use clone::Clone;
use iter::{DoubleEndedIterator, Iterator};
use num::CheckedMul;
use container::Container;
use mem::{size_of, move_val_init};
use cast::{forget, transmute};
use rt::global_heap::{malloc_raw, realloc_raw};
use vec::{ImmutableVector, Items, MutableVector};
use unstable::raw::Slice;
use ptr::{offset, read_ptr};
use libc::{free, c_void};

pub struct Vec<T> {
    priv len: uint,
    priv cap: uint,
    priv ptr: *mut T
}

impl<T> Vec<T> {
    #[inline]
    pub fn new() -> Vec<T> {
        Vec { len: 0, cap: 0, ptr: 0 as *mut T }
    }

    pub fn with_capacity(capacity: uint) -> Vec<T> {
        if capacity == 0 {
            Vec::new()
        } else {
            let size = capacity.checked_mul(&size_of::<T>()).expect("capacity overflow");
            let ptr = unsafe { malloc_raw(size) };
            Vec { len: 0, cap: capacity, ptr: ptr as *mut T }
        }
    }

    pub fn from_fn(length: uint, op: |uint| -> T) -> Vec<T> {
        unsafe {
            let mut xs = Vec::with_capacity(length);
            while xs.len < length {
                move_val_init(xs.as_mut_slice().unsafe_mut_ref(xs.len), op(xs.len));
                xs.len += 1;
            }
            xs
        }
    }
}

impl<T: Clone> Vec<T> {
    pub fn from_elem(length: uint, value: T) -> Vec<T> {
        unsafe {
            let mut xs = Vec::with_capacity(length);
            while xs.len < length {
                move_val_init(xs.as_mut_slice().unsafe_mut_ref(xs.len), value.clone());
                xs.len += 1;
            }
            xs
        }
    }
}

impl<T> Container for Vec<T> {
    #[inline]
    fn len(&self) -> uint {
        self.len
    }
}

impl<T> Vec<T> {
    #[inline]
    pub fn capacity(&self) -> uint {
        self.cap
    }

    pub fn reserve_exact(&mut self, capacity: uint) {
        if capacity >= self.len {
            let size = capacity.checked_mul(&size_of::<T>()).expect("capacity overflow");
            self.cap = capacity;
            unsafe {
                self.ptr = realloc_raw(self.ptr as *mut u8, size) as *mut T;
            }
        }
    }

    pub fn shrink_to_fit(&mut self) {
        if self.len == 0 {
            unsafe { free(self.ptr as *mut c_void) };
            self.cap = 0;
            self.ptr = 0 as *mut T;
        } else {
            unsafe {
                // Overflow check is unnecessary as the vector is already at least this large.
                self.ptr = realloc_raw(self.ptr as *mut u8, self.len * size_of::<T>()) as *mut T;
            }
            self.cap = self.len;
        }
    }

    #[inline]
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
            if old_size > size { fail!("capacity overflow") }
            unsafe {
                self.ptr = realloc_raw(self.ptr as *mut u8, size) as *mut T;
            }
        }

        unsafe {
            let end = offset(self.ptr as *T, self.len as int) as *mut T;
            move_val_init(&mut *end, value);
            self.len += 1;
        }
    }

    pub fn truncate(&mut self, len: uint) {
        unsafe {
            let mut i = len;
            // drop any extra elements
            while i < self.len {
                read_ptr(self.as_slice().unsafe_ref(i));
                i += 1;
            }
        }
        self.len = len;
    }

    #[inline]
    pub fn as_slice<'a>(&'a self) -> &'a [T] {
        let slice = Slice { data: self.ptr as *T, len: self.len };
        unsafe { transmute(slice) }
    }

    #[inline]
    pub fn as_mut_slice<'a>(&'a mut self) -> &'a mut [T] {
        let slice = Slice { data: self.ptr as *T, len: self.len };
        unsafe { transmute(slice) }
    }

    #[inline]
    pub fn move_iter(self) -> MoveItems<T> {
        unsafe {
            let iter = transmute(self.as_slice().iter());
            let ptr = self.ptr as *mut c_void;
            forget(self);
            MoveItems { allocation: ptr, iter: iter }
        }
    }

    #[inline]
    pub unsafe fn set_len(&mut self, len: uint) {
        self.len = len;
    }
}


#[unsafe_destructor]
impl<T> Drop for Vec<T> {
    fn drop(&mut self) {
        unsafe {
            for x in self.as_mut_slice().iter() {
                read_ptr(x);
            }
            free(self.ptr as *mut c_void)
        }
    }
}

pub struct MoveItems<T> {
    priv allocation: *mut c_void, // the block of memory allocated for the vector
    priv iter: Items<'static, T>
}

impl<T> Iterator<T> for MoveItems<T> {
    #[inline]
    fn next(&mut self) -> Option<T> {
        unsafe {
            self.iter.next().map(|x| read_ptr(x))
        }
    }

    #[inline]
    fn size_hint(&self) -> (uint, Option<uint>) {
        self.iter.size_hint()
    }
}

impl<T> DoubleEndedIterator<T> for MoveItems<T> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        unsafe {
            self.iter.next_back().map(|x| read_ptr(x))
        }
    }
}

#[unsafe_destructor]
impl<T> Drop for MoveItems<T> {
    fn drop(&mut self) {
        // destroy the remaining elements
        for _x in *self {}
        unsafe {
            free(self.allocation)
        }
    }
}
