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

impl<T> Deref<T> for Rc<T> {
    fn deref(&self) -> &T {
        unsafe { &*self.value }
    }
}

struct Point {
    x: int,
    y: int
}

impl Point {
    fn get(&self) -> (int, int) {
        (self.x, self.y)
    }

    fn set(&mut self, x: int, y: int) {
        self.x = x;
        self.y = y;
    }

    fn x_ref(&self) -> &int {
        &self.x
    }

    fn y_mut(&mut self) -> &mut int {
        &mut self.y
    }
}

fn deref_imm_field(x: Rc<Point>) {
    let _i = &x.y;
}

fn deref_mut_field1(x: Rc<Point>) {
    let _i = &mut x.y; //~ ERROR cannot borrow
}

fn deref_mut_field2(mut x: Rc<Point>) {
    let _i = &mut x.y; //~ ERROR cannot borrow
}

fn deref_extend_field(x: &Rc<Point>) -> &int {
    &x.y
}

fn deref_extend_mut_field1(x: &Rc<Point>) -> &mut int {
    &mut x.y //~ ERROR cannot borrow
}

fn deref_extend_mut_field2(x: &mut Rc<Point>) -> &mut int {
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
    let _i = x.get();
}

fn deref_mut_method1(x: Rc<Point>) {
    x.set(0, 0); //~ ERROR cannot borrow
}

fn deref_mut_method2(mut x: Rc<Point>) {
    x.set(0, 0); //~ ERROR cannot borrow
}

fn deref_extend_method(x: &Rc<Point>) -> &int {
    x.x_ref()
}

fn deref_extend_mut_method1(x: &Rc<Point>) -> &mut int {
    x.y_mut() //~ ERROR cannot borrow
}

fn deref_extend_mut_method2(x: &mut Rc<Point>) -> &mut int {
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
