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

impl<T> Deref for Own<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*self.value }
    }
}

impl<T> DerefMut for Own<T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *self.value }
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

fn deref_imm_field(x: Own<Point>) {
    let __isize = &x.y;
}

fn deref_mut_field1(x: Own<Point>) {
    let __isize = &mut x.y; //~ ERROR cannot borrow
}

fn deref_mut_field2(mut x: Own<Point>) {
    let __isize = &mut x.y;
}

fn deref_extend_field(x: &Own<Point>) -> &isize {
    &x.y
}

fn deref_extend_mut_field1(x: &Own<Point>) -> &mut isize {
    &mut x.y //~ ERROR cannot borrow
}

fn deref_extend_mut_field2(x: &mut Own<Point>) -> &mut isize {
    &mut x.y
}

fn deref_extend_mut_field3(x: &mut Own<Point>) {
    // Hmm, this is unfortunate, because with box it would work,
    // but it's presently the expected outcome. See `deref_extend_mut_field4`
    // for the workaround.

    let _x = &mut x.x;
    let _y = &mut x.y; //~ ERROR cannot borrow
}

fn deref_extend_mut_field4<'a>(x: &'a mut Own<Point>) {
    let p = &mut **x;
    let _x = &mut p.x;
    let _y = &mut p.y;
}

fn assign_field1<'a>(x: Own<Point>) {
    x.y = 3; //~ ERROR cannot borrow
}

fn assign_field2<'a>(x: &'a Own<Point>) {
    x.y = 3; //~ ERROR cannot assign
}

fn assign_field3<'a>(x: &'a mut Own<Point>) {
    x.y = 3;
}

fn assign_field4<'a>(x: &'a mut Own<Point>) {
    let _p: &mut Point = &mut **x;
    x.y = 3; //~ ERROR cannot borrow
}

// FIXME(eddyb) #12825 This shouldn't attempt to call deref_mut.
/*
fn deref_imm_method(x: Own<Point>) {
    let __isize = x.get();
}
*/

fn deref_mut_method1(x: Own<Point>) {
    x.set(0, 0); //~ ERROR cannot borrow
}

fn deref_mut_method2(mut x: Own<Point>) {
    x.set(0, 0);
}

fn deref_extend_method(x: &Own<Point>) -> &isize {
    x.x_ref()
}

fn deref_extend_mut_method1(x: &Own<Point>) -> &mut isize {
    x.y_mut() //~ ERROR cannot borrow
}

fn deref_extend_mut_method2(x: &mut Own<Point>) -> &mut isize {
    x.y_mut()
}

fn assign_method1<'a>(x: Own<Point>) {
    *x.y_mut() = 3; //~ ERROR cannot borrow
}

fn assign_method2<'a>(x: &'a Own<Point>) {
    *x.y_mut() = 3; //~ ERROR cannot borrow
}

fn assign_method3<'a>(x: &'a mut Own<Point>) {
    *x.y_mut() = 3;
}

pub fn main() {}
