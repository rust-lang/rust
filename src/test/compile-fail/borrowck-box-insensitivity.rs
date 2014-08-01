// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct A {
    x: Box<int>,
    y: int,
}

struct B {
    x: Box<int>,
    y: Box<int>,
}

struct C {
    x: Box<A>,
    y: int,
}

struct D {
    x: Box<A>,
    y: Box<int>,
}

fn copy_after_move() {
    let a = box A { x: box 0, y: 1 };
    let _x = a.x;
    let _y = a.y; //~ ERROR use of partially moved
}

fn move_after_move() {
    let a = box B { x: box 0, y: box 1 };
    let _x = a.x;
    let _y = a.y; //~ ERROR use of partially moved
}

fn borrow_after_move() {
    let a = box A { x: box 0, y: 1 };
    let _x = a.x;
    let _y = &a.y; //~ ERROR use of partially moved
}

fn move_after_borrow() {
    let a = box B { x: box 0, y: box 1 };
    let _x = &a.x;
    let _y = a.y; //~ ERROR cannot move
}

fn copy_after_mut_borrow() {
    let mut a = box A { x: box 0, y: 1 };
    let _x = &mut a.x;
    let _y = a.y; //~ ERROR cannot use
}

fn move_after_mut_borrow() {
    let mut a = box B { x: box 0, y: box 1 };
    let _x = &mut a.x;
    let _y = a.y; //~ ERROR cannot move
}

fn borrow_after_mut_borrow() {
    let mut a = box A { x: box 0, y: 1 };
    let _x = &mut a.x;
    let _y = &a.y; //~ ERROR cannot borrow
}

fn mut_borrow_after_borrow() {
    let mut a = box A { x: box 0, y: 1 };
    let _x = &a.x;
    let _y = &mut a.y; //~ ERROR cannot borrow
}

fn copy_after_move_nested() {
    let a = box C { x: box A { x: box 0, y: 1 }, y: 2 };
    let _x = a.x.x;
    let _y = a.y; //~ ERROR use of partially moved
}

fn move_after_move_nested() {
    let a = box D { x: box A { x: box 0, y: 1 }, y: box 2 };
    let _x = a.x.x;
    let _y = a.y; //~ ERROR use of partially moved
}

fn borrow_after_move_nested() {
    let a = box C { x: box A { x: box 0, y: 1 }, y: 2 };
    let _x = a.x.x;
    let _y = &a.y; //~ ERROR use of partially moved
}

fn move_after_borrow_nested() {
    let a = box D { x: box A { x: box 0, y: 1 }, y: box 2 };
    let _x = &a.x.x;
    let _y = a.y; //~ ERROR cannot move
}

fn copy_after_mut_borrow_nested() {
    let mut a = box C { x: box A { x: box 0, y: 1 }, y: 2 };
    let _x = &mut a.x.x;
    let _y = a.y; //~ ERROR cannot use
}

fn move_after_mut_borrow_nested() {
    let mut a = box D { x: box A { x: box 0, y: 1 }, y: box 2 };
    let _x = &mut a.x.x;
    let _y = a.y; //~ ERROR cannot move
}

fn borrow_after_mut_borrow_nested() {
    let mut a = box C { x: box A { x: box 0, y: 1 }, y: 2 };
    let _x = &mut a.x.x;
    let _y = &a.y; //~ ERROR cannot borrow
}

fn mut_borrow_after_borrow_nested() {
    let mut a = box C { x: box A { x: box 0, y: 1 }, y: 2 };
    let _x = &a.x.x;
    let _y = &mut a.y; //~ ERROR cannot borrow
}

fn main() {
    copy_after_move();
    move_after_move();
    borrow_after_move();

    move_after_borrow();

    copy_after_mut_borrow();
    move_after_mut_borrow();
    borrow_after_mut_borrow();
    mut_borrow_after_borrow();

    copy_after_move_nested();
    move_after_move_nested();
    borrow_after_move_nested();

    move_after_borrow_nested();

    copy_after_mut_borrow_nested();
    move_after_mut_borrow_nested();
    borrow_after_mut_borrow_nested();
    mut_borrow_after_borrow_nested();
}

