// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test illustrates use of `box (<place>) <value>` syntax to
// initialize into the end of a Vec<T>.

#![feature(unsafe_destructor)]

use std::cell::{UnsafeCell};
use std::ops::{Placer,PlacementAgent};

struct EmplaceBackPlacer<'a, T:'a> {
    vec: &'a mut Vec<T>,
}

trait EmplaceBack<'a, T> {
    fn emplace_back(&'a mut self) -> EmplaceBackPlacer<'a, T>;
}

impl<'a, T:'a> EmplaceBack<'a, T> for Vec<T> {
    fn emplace_back(&'a mut self) -> EmplaceBackPlacer<'a, T> {
        EmplaceBackPlacer { vec: self }
    }
}

pub fn main() {
    let mut v : Vec<[f32, ..4]> = vec![];
    v.push([10., 20., 30., 40.]);
    v.push([11., 21., 31., 41.]);
    let () = // (Explicitly showing `box` returns `()` here.)
        box (v.emplace_back()) [12., 22., 32., 42.];
    assert!(same_contents(
        v.as_slice(),
        [[10., 20., 30., 40.],
         [11., 21., 31., 41.],
         [12., 22., 32., 42.],
         ]));
}

fn same_contents<T:PartialEq>(a: &[[T, ..4]], b: &[[T, ..4]]) -> bool {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    for i in range(0, len) {
        if a[i].as_slice() != b[i].as_slice() {
            return false;
        }
    }
    return true;
}

struct EmplaceBackAgent<T> {
    vec_ptr: *mut Vec<T>,
    offset: uint,
}

impl<'a, T> Placer<T, (), EmplaceBackAgent<T>> for EmplaceBackPlacer<'a, T> {
    fn make_place(&mut self) -> EmplaceBackAgent<T> {
        let len = self.vec.len();
        let v = self.vec as *mut Vec<T>;
        unsafe {
            (*v).reserve_additional(1);
        }
        EmplaceBackAgent { vec_ptr: v, offset: len }
    }
}

impl<T> PlacementAgent<T, ()> for EmplaceBackAgent<T> {
    unsafe fn pointer(&self) -> *mut T {
        assert_eq!((*self.vec_ptr).len(), self.offset);
        assert!(self.offset < (*self.vec_ptr).capacity());
        (*self.vec_ptr).as_mut_ptr().offset(self.offset.to_int().unwrap())
    }

    unsafe fn finalize(self) -> () {
        assert_eq!((*self.vec_ptr).len(), self.offset);
        assert!(self.offset < (*self.vec_ptr).capacity());
        (*self.vec_ptr).set_len(self.offset + 1);
    }
}

#[unsafe_destructor]
impl<T> Drop for EmplaceBackAgent<T> {
    fn drop(&mut self) {
        // Do not need to do anything; all `make_place` did was ensure
        // we had some space reserved, it did not touch the state of
        // the vector itself.
    }
}
