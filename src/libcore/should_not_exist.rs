// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// As noted by this file name, this file should not exist. This file should not
// exist because it performs allocations which libcore is not allowed to do. The
// reason for this file's existence is that the `~[T]` type is a language-
// defined type. Traits are defined in libcore, such as `Clone`, which these
// types need to implement, but the implementation can only be found in
// libcore.
//
// Plan of attack for solving this problem:
//
//      1. Implement DST
//      2. Make `Box<T>` not a language feature
//      3. Move `Box<T>` to a separate crate, liballoc.
//      4. Implement relevant traits in liballoc, not libcore
//
// Currently, no progress has been made on this list.

use clone::Clone;
use container::Container;
use finally::try_finally;
use intrinsics;
use iter::{range, Iterator};
use mem;
use num::{CheckedMul, CheckedAdd};
use option::{Some, None};
use ptr::RawPtr;
use ptr;
use raw::Vec;
use slice::ImmutableVector;

#[allow(ctypes)]
extern {
    fn rust_allocate(size: uint, align: uint) -> *u8;
    fn rust_deallocate(ptr: *u8, size: uint, align: uint);
}

unsafe fn alloc(cap: uint) -> *mut Vec<()> {
    let cap = cap.checked_add(&mem::size_of::<Vec<()>>()).unwrap();
    // this should use the real alignment, but the new representation will take care of that
    let ret = rust_allocate(cap, 8) as *mut Vec<()>;
    if ret.is_null() {
        intrinsics::abort();
    }
    (*ret).fill = 0;
    (*ret).alloc = cap;
    ret
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

            let a_size = mem::size_of::<A>();
            let a_size = if a_size == 0 {1} else {a_size};
            (*ret).fill = len * a_size;
            (*ret).alloc = len * a_size;

            let mut i = 0;
            let p = &mut (*ret).data as *mut _ as *mut A;
            try_finally(
                &mut i, (),
                |i, ()| while *i < len {
                    ptr::write(
                        &mut(*p.offset(*i as int)),
                        self.unsafe_ref(*i).clone());
                    *i += 1;
                },
                |i| if *i < len {
                    // we must be failing, clean up after ourselves
                    for j in range(0, *i as int) {
                        ptr::read(&*p.offset(j));
                    }
                    rust_deallocate(ret as *u8, 0, 8);
                });
            mem::transmute(ret)
        }
    }
}
