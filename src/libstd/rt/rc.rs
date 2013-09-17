// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! An owned, task-local, reference counted type
//!
//! # Safety note
//!
//! XXX There is currently no type-system mechanism for enforcing that
//! reference counted types are both allocated on the exchange heap
//! and also non-sendable
//!
//! This doesn't prevent borrowing multiple aliasable mutable pointers

use ops::Drop;
use clone::Clone;
use libc::c_void;
use cast;

pub struct RC<T> {
    p: *c_void // ~(uint, T)
}

impl<T> RC<T> {
    pub fn new(val: T) -> RC<T> {
        unsafe {
            let v = ~(1, val);
            let p: *c_void = cast::transmute(v);
            RC { p: p }
        }
    }

    fn get_mut_state(&mut self) -> *mut (uint, T) {
        unsafe {
            let p: &mut ~(uint, T) = cast::transmute(&mut self.p);
            let p: *mut (uint, T) = &mut **p;
            return p;
        }
    }

    fn get_state(&self) -> *(uint, T) {
        unsafe {
            let p: &~(uint, T) = cast::transmute(&self.p);
            let p: *(uint, T) = &**p;
            return p;
        }
    }

    pub fn unsafe_borrow_mut(&mut self) -> *mut T {
        unsafe {
            match *self.get_mut_state() {
                (_, ref mut p) => {
                    let p: *mut T = p;
                    return p;
                }
            }
        }
    }

    pub fn refcount(&self) -> uint {
        unsafe {
            match *self.get_state() {
                (count, _) => count
            }
        }
    }
}

#[unsafe_destructor]
impl<T> Drop for RC<T> {
    fn drop(&mut self) {
        assert!(self.refcount() > 0);

        unsafe {
            match *self.get_mut_state() {
                (ref mut count, _) => {
                    *count = *count - 1
                }
            }

            if self.refcount() == 0 {
                let _: ~(uint, T) = cast::transmute(self.p);
            }
        }
    }
}

impl<T> Clone for RC<T> {
    fn clone(&self) -> RC<T> {
        unsafe {
            // XXX: Mutable clone
            let this: &mut RC<T> = cast::transmute_mut(self);

            match *this.get_mut_state() {
                (ref mut count, _) => {
                    *count = *count + 1;
                }
            }
        }

        RC { p: self.p }
    }
}

#[cfg(test)]
mod test {
    use super::RC;

    #[test]
    fn smoke_test() {
        unsafe {
            let mut v1 = RC::new(100);
            assert!(*v1.unsafe_borrow_mut() == 100);
            assert!(v1.refcount() == 1);

            let mut v2 = v1.clone();
            assert!(*v2.unsafe_borrow_mut() == 100);
            assert!(v2.refcount() == 2);

            *v2.unsafe_borrow_mut() = 200;
            assert!(*v2.unsafe_borrow_mut() == 200);
            assert!(*v1.unsafe_borrow_mut() == 200);

            let v3 = v2.clone();
            assert!(v3.refcount() == 3);
            {
                let _v1 = v1;
                let _v2 = v2;
            }
            assert!(v3.refcount() == 1);
        }
    }
}
