// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]

use std::thread::Thread;

fn borrow<T>(_: &T) { }

fn different_vars_after_borrows() {
    let x1 = box 1is;
    let p1 = &x1;
    let x2 = box 2is;
    let p2 = &x2;
    Thread::spawn(move|| {
        drop(x1); //~ ERROR cannot move `x1` into closure because it is borrowed
        drop(x2); //~ ERROR cannot move `x2` into closure because it is borrowed
    });
    borrow(&*p1);
    borrow(&*p2);
}

fn different_vars_after_moves() {
    let x1 = box 1is;
    drop(x1);
    let x2 = box 2is;
    drop(x2);
    Thread::spawn(move|| {
        drop(x1); //~ ERROR capture of moved value: `x1`
        drop(x2); //~ ERROR capture of moved value: `x2`
    });
}

fn same_var_after_borrow() {
    let x = box 1is;
    let p = &x;
    Thread::spawn(move|| {
        drop(x); //~ ERROR cannot move `x` into closure because it is borrowed
        drop(x); //~ ERROR use of moved value: `x`
    });
    borrow(&*p);
}

fn same_var_after_move() {
    let x = box 1is;
    drop(x);
    Thread::spawn(move|| {
        drop(x); //~ ERROR capture of moved value: `x`
        drop(x); //~ ERROR use of moved value: `x`
    });
}

fn main() {
    different_vars_after_borrows();
    different_vars_after_moves();
    same_var_after_borrow();
    same_var_after_move();
}

