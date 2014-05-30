// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This tests that we can't modify Box<&mut T> contents while they
// are borrowed.

struct A { a: int }
struct B<'a> { a: Box<&'a mut int> }

fn borrow_in_var_from_var() {
    let mut x: int = 1;
    let y = box &mut x;
    let p = &y;
    let q = &***p;
    **y = 2; //~ ERROR cannot assign to `**y` because it is borrowed
    drop(p);
    drop(q);
}

fn borrow_in_var_from_field() {
    let mut x = A { a: 1 };
    let y = box &mut x.a;
    let p = &y;
    let q = &***p;
    **y = 2; //~ ERROR cannot assign to `**y` because it is borrowed
    drop(p);
    drop(q);
}

fn borrow_in_field_from_var() {
    let mut x: int = 1;
    let y = B { a: box &mut x };
    let p = &y.a;
    let q = &***p;
    **y.a = 2; //~ ERROR cannot assign to `**y.a` because it is borrowed
    drop(p);
    drop(q);
}

fn borrow_in_field_from_field() {
    let mut x = A { a: 1 };
    let y = B { a: box &mut x.a };
    let p = &y.a;
    let q = &***p;
    **y.a = 2; //~ ERROR cannot assign to `**y.a` because it is borrowed
    drop(p);
    drop(q);
}

fn main() {
    borrow_in_var_from_var();
    borrow_in_var_from_field();
    borrow_in_field_from_var();
    borrow_in_field_from_field();
}

