// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test how overloaded deref interacts with borrows when DerefMut
// is implemented.

use std::ops::{Deref, DerefMut};

struct Own<T> {
    value: *mut T
}

impl<T> Deref<T> for Own<T> {
    fn deref<'a>(&'a self) -> &'a T {
        unsafe { &*self.value }
    }
}

impl<T> DerefMut<T> for Own<T> {
    fn deref_mut<'a>(&'a mut self) -> &'a mut T {
        unsafe { &mut *self.value }
    }
}

fn deref_imm(x: Own<int>) {
    let _i = &*x;
}

fn deref_mut1(x: Own<int>) {
    let _i = &mut *x; //~ ERROR cannot borrow
}

fn deref_mut2(mut x: Own<int>) {
    let _i = &mut *x;
}

fn deref_extend<'a>(x: &'a Own<int>) -> &'a int {
    &**x
}

fn deref_extend_mut1<'a>(x: &'a Own<int>) -> &'a mut int {
    &mut **x //~ ERROR cannot borrow
}

fn deref_extend_mut2<'a>(x: &'a mut Own<int>) -> &'a mut int {
    &mut **x
}

fn assign1<'a>(x: Own<int>) {
    *x = 3; //~ ERROR cannot borrow
}

fn assign2<'a>(x: &'a Own<int>) {
    **x = 3; //~ ERROR cannot borrow
}

fn assign3<'a>(x: &'a mut Own<int>) {
    **x = 3;
}

pub fn main() {}