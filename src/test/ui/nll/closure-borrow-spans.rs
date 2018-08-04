// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// check that existing borrows due to a closure capture give a special note

#![feature(nll)]

fn move_while_borrowed(x: String) {
    let f = || x.len();
    let y = x; //~ ERROR
    f.use_ref();
}

fn borrow_mut_while_borrowed(mut x: i32) {
    let f = || x;
    let y = &mut x; //~ ERROR
    f.use_ref();
}

fn drop_while_borrowed() {
    let f;
    {
        let x = 1;
        f = || x; //~ ERROR
    }
    f.use_ref();
}

fn assign_while_borrowed(mut x: i32) {
    let f = || x;
    x = 1; //~ ERROR
    f.use_ref();
}

fn copy_while_borrowed_mut(mut x: i32) {
    let f = || x = 0;
    let y = x; //~ ERROR
    f.use_ref();
}

fn borrow_while_borrowed_mut(mut x: i32) {
    let f = || x = 0;
    let y = &x; //~ ERROR
    f.use_ref();
}

fn borrow_mut_while_borrowed_mut(mut x: i32) {
    let f = || x = 0;
    let y = &mut x; //~ ERROR
    f.use_ref();
}

fn drop_while_borrowed_mut() {
    let f;
    {
        let mut x = 1;
        f = || x = 0; //~ ERROR
    }
    f.use_ref();
}

fn assign_while_borrowed_mut(mut x: i32) {
    let f = || x = 0;
    x = 1; //~ ERROR
    f.use_ref();
}

fn copy_while_borrowed_unique(x: &mut i32) {
    let f = || *x = 0;
    let y = x; //~ ERROR
    f.use_ref();
}

fn borrow_while_borrowed_unique(x: &mut i32) {
    let f = || *x = 0;
    let y = &x; //~ ERROR
    f.use_ref();
}

fn borrow_mut_while_borrowed_unique(mut x: &mut i32) {
    let f = || *x = 0;
    let y = &mut x; //~ ERROR
    f.use_ref();
}

fn drop_while_borrowed_unique() {
    let mut z = 1;
    let f;
    {
        let x = &mut z;
        f = || *x = 0; //~ ERROR
    }
    f.use_ref();
}

fn assign_while_borrowed_unique(x: &mut i32) {
    let f = || *x = 0;
    *x = 1; //~ ERROR
    f.use_ref();
}

fn main() {}

trait Fake { fn use_mut(&mut self) { } fn use_ref(&self) { }  }
impl<T> Fake for T { }
