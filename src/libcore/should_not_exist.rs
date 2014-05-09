// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use char::Char;
use clone::Clone;
use container::Container;
use default::Default;
use finally::try_finally;
use intrinsics;
use iter::{range, Iterator, FromIterator};
use mem;
use num::{CheckedMul, CheckedAdd};
use option::{Some, None};
use ptr::RawPtr;
use ptr;
use raw::Vec;
use slice::ImmutableVector;
use str::StrSlice;

#[cfg(not(test))] use ops::Add;

#[allow(ctypes)]
extern {
    #[cfg(stage0)]
    fn rust_malloc(size: uint) -> *u8;
    #[cfg(not(stage0))]
    fn rust_malloc(size: uint, align: uint) -> *u8;
    fn rust_free(ptr: *u8, size: uint, align: uint);
}

#[cfg(stage0)]
unsafe fn alloc(cap: uint) -> *mut Vec<()> {
    let cap = cap.checked_add(&mem::size_of::<Vec<()>>()).unwrap();
    let ret = rust_malloc(cap) as *mut Vec<()>;
    if ret.is_null() {
        intrinsics::abort();
    }
    (*ret).fill = 0;
    (*ret).alloc = cap;
    ret
}

#[cfg(not(stage0))]
unsafe fn alloc(cap: uint) -> *mut Vec<()> {
    let cap = cap.checked_add(&mem::size_of::<Vec<()>>()).unwrap();
    // this should use the real alignment, but the new representation will take care of that
    let ret = rust_malloc(cap, 8) as *mut Vec<()>;
    if ret.is_null() {
        intrinsics::abort();
    }
    (*ret).fill = 0;
    (*ret).alloc = cap;
    ret
}

// Strings

impl Default for ~str {
    fn default() -> ~str {
        unsafe {
            // Get some memory
            let ptr = alloc(0);

            // Initialize the memory
            (*ptr).fill = 0;
            (*ptr).alloc = 0;

            mem::transmute(ptr)
        }
    }
}

impl Clone for ~str {
    fn clone(&self) -> ~str {
        // Don't use the clone() implementation above because it'll start
        // requiring the eh_personality lang item (no fun)
        unsafe {
            let bytes = self.as_bytes().as_ptr();
            let len = self.len();

            let ptr = alloc(len) as *mut Vec<u8>;
            ptr::copy_nonoverlapping_memory(&mut (*ptr).data, bytes, len);
            (*ptr).fill = len;
            (*ptr).alloc = len;

            mem::transmute(ptr)
        }
    }
}

impl FromIterator<char> for ~str {
    #[inline]
    fn from_iter<T: Iterator<char>>(mut iterator: T) -> ~str {
        let (lower, _) = iterator.size_hint();
        let mut cap = if lower == 0 {16} else {lower};
        let mut len = 0;
        let mut tmp = [0u8, ..4];

        unsafe {
            let mut ptr = alloc(cap) as *mut Vec<u8>;
            let mut ret = mem::transmute(ptr);
            for ch in iterator {
                let amt = ch.encode_utf8(tmp);

                if len + amt > cap {
                    cap = cap.checked_mul(&2).unwrap();
                    if cap < len + amt {
                        cap = len + amt;
                    }
                    let ptr2 = alloc(cap) as *mut Vec<u8>;
                    ptr::copy_nonoverlapping_memory(&mut (*ptr2).data,
                                                    &(*ptr).data,
                                                    len);
                    // FIXME: #13994: port to the sized deallocation API when available
                    rust_free(ptr as *u8, 0, 8);
                    mem::forget(ret);
                    ret = mem::transmute(ptr2);
                    ptr = ptr2;
                }

                let base = &mut (*ptr).data as *mut u8;
                for byte in tmp.slice_to(amt).iter() {
                    *base.offset(len as int) = *byte;
                    len += 1;
                }
                (*ptr).fill = len;
            }
            ret
        }
    }
}

#[cfg(not(test))]
impl<'a> Add<&'a str,~str> for &'a str {
    #[inline]
    fn add(&self, rhs: & &'a str) -> ~str {
        let amt = self.len().checked_add(&rhs.len()).unwrap();
        unsafe {
            let ptr = alloc(amt) as *mut Vec<u8>;
            let base = &mut (*ptr).data as *mut _;
            ptr::copy_nonoverlapping_memory(base,
                                            self.as_bytes().as_ptr(),
                                            self.len());
            let base = base.offset(self.len() as int);
            ptr::copy_nonoverlapping_memory(base,
                                            rhs.as_bytes().as_ptr(),
                                            rhs.len());
            (*ptr).fill = amt;
            (*ptr).alloc = amt;
            mem::transmute(ptr)
        }
    }
}

// Arrays

impl<A: Clone> Clone for ~[A] {
    #[inline]
    fn clone(&self) -> ~[A] {
        let len = self.len();
        let data_size = len.checked_mul(&mem::size_of::<A>()).unwrap();
        let size = mem::size_of::<Vec<()>>().checked_add(&data_size).unwrap();

        unsafe {
            let ret = alloc(size) as *mut Vec<A>;

            (*ret).fill = len * mem::nonzero_size_of::<A>();
            (*ret).alloc = len * mem::nonzero_size_of::<A>();

            let mut i = 0;
            let p = &mut (*ret).data as *mut _ as *mut A;
            try_finally(
                &mut i, (),
                |i, ()| while *i < len {
                    mem::move_val_init(
                        &mut(*p.offset(*i as int)),
                        self.unsafe_ref(*i).clone());
                    *i += 1;
                },
                |i| if *i < len {
                    // we must be failing, clean up after ourselves
                    for j in range(0, *i as int) {
                        ptr::read(&*p.offset(j));
                    }
                    rust_free(ret as *u8, 0, 8);
                });
            mem::transmute(ret)
        }
    }
}
