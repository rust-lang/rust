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
    value: *const T
}

impl<T> Deref for Rc<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.value }
    }
}

struct Point {
    x: isize,
    y: isize
}

impl Point {
    fn get(&self) -> (isize, isize) {
        (self.x, self.y)
    }

    fn set(&mut self, x: isize, y: isize) {
        self.x = x;
        self.y = y;
    }

    fn x_ref(&self) -> &isize {
        &self.x
    }

    fn y_mut(&mut self) -> &mut isize {
        &mut self.y
    }
}

fn deref_imm_field(x: Rc<Point>) {
    let __isize = &x.y;
}

fn deref_mut_field1(x: Rc<Point>) {
    let __isize = &mut x.y; //~ ERROR cannot borrow
}

fn deref_mut_field2(mut x: Rc<Point>) {
    let __isize = &mut x.y; //~ ERROR cannot borrow
}

fn deref_extend_field(x: &Rc<Point>) -> &isize {
    &x.y
}

fn deref_extend_mut_field1(x: &Rc<Point>) -> &mut isize {
    &mut x.y //~ ERROR cannot borrow
}

fn deref_extend_mut_field2(x: &mut Rc<Point>) -> &mut isize {
    &mut x.y //~ ERROR cannot borrow
}

fn assign_field1<'a>(x: Rc<Point>) {
    x.y = 3; //~ ERROR cannot assign
}

fn assign_field2<'a>(x: &'a Rc<Point>) {
    x.y = 3; //~ ERROR cannot assign
}

fn assign_field3<'a>(x: &'a mut Rc<Point>) {
    x.y = 3; //~ ERROR cannot assign
}

fn deref_imm_method(x: Rc<Point>) {
    let __isize = x.get();
}

fn deref_mut_method1(x: Rc<Point>) {
    x.set(0, 0); //~ ERROR cannot borrow
}

fn deref_mut_method2(mut x: Rc<Point>) {
    x.set(0, 0); //~ ERROR cannot borrow
}

fn deref_extend_method(x: &Rc<Point>) -> &isize {
    x.x_ref()
}

fn deref_extend_mut_method1(x: &Rc<Point>) -> &mut isize {
    x.y_mut() //~ ERROR cannot borrow
}

fn deref_extend_mut_method2(x: &mut Rc<Point>) -> &mut isize {
    x.y_mut() //~ ERROR cannot borrow
}

fn assign_method1<'a>(x: Rc<Point>) {
    *x.y_mut() = 3; //~ ERROR cannot borrow
}

fn assign_method2<'a>(x: &'a Rc<Point>) {
    *x.y_mut() = 3; //~ ERROR cannot borrow
}

fn assign_method3<'a>(x: &'a mut Rc<Point>) {
    *x.y_mut() = 3; //~ ERROR cannot borrow
}

pub fn main() {}
