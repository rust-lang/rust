// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test how overloaded deref interacts with borrows when only
// Deref and not DerefMut is implemented.

use std::ops::Deref;

struct Rc<T> {
    value: *T
}

impl<T> Deref<T> for Rc<T> {
    fn deref<'a>(&'a self) -> &'a T {
        unsafe { &*self.value }
    }
}

fn deref_imm(x: Rc<int>) {
    let _i = &*x;
}

fn deref_mut1(x: Rc<int>) {
    let _i = &mut *x; //~ ERROR cannot borrow
}

fn deref_mut2(mut x: Rc<int>) {
    let _i = &mut *x; //~ ERROR cannot borrow
}

fn deref_extend<'a>(x: &'a Rc<int>) -> &'a int {
    &**x
}

fn deref_extend_mut1<'a>(x: &'a Rc<int>) -> &'a mut int {
    &mut **x //~ ERROR cannot borrow
}

fn deref_extend_mut2<'a>(x: &'a mut Rc<int>) -> &'a mut int {
    &mut **x //~ ERROR cannot borrow
}

fn assign1<'a>(x: Rc<int>) {
    *x = 3; //~ ERROR cannot assign
}

fn assign2<'a>(x: &'a Rc<int>) {
    **x = 3; //~ ERROR cannot assign
}

fn assign3<'a>(x: &'a mut Rc<int>) {
    **x = 3; //~ ERROR cannot assign
}

pub fn main() {}
