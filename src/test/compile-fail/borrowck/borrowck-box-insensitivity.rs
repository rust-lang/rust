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

struct A {
    x: Box<isize>,
    y: isize,
}

struct B {
    x: Box<isize>,
    y: Box<isize>,
}

struct C {
    x: Box<A>,
    y: isize,
}

struct D {
    x: Box<A>,
    y: Box<isize>,
}

fn copy_after_move() {
    let a: Box<_> = box A { x: box 0, y: 1 };
    let _x = a.x;
    let _y = a.y; //~ ERROR use of moved
    //~^^ NOTE `a` moved here (through moving `a.x`)
}

fn move_after_move() {
    let a: Box<_> = box B { x: box 0, y: box 1 };
    let _x = a.x;
    let _y = a.y; //~ ERROR use of moved
    //~^^ NOTE `a` moved here (through moving `a.x`)
}

fn borrow_after_move() {
    let a: Box<_> = box A { x: box 0, y: 1 };
    let _x = a.x;
    let _y = &a.y; //~ ERROR use of moved
    //~^^ NOTE `a` moved here (through moving `a.x`)
}

fn move_after_borrow() {
    let a: Box<_> = box B { x: box 0, y: box 1 };
    let _x = &a.x;
    //~^ NOTE borrow of `a.x` occurs here
    let _y = a.y; //~ ERROR cannot move
}

fn copy_after_mut_borrow() {
    let mut a: Box<_> = box A { x: box 0, y: 1 };
    let _x = &mut a.x;
    //~^ NOTE borrow of `a.x` occurs here
    let _y = a.y; //~ ERROR cannot use
}

fn move_after_mut_borrow() {
    let mut a: Box<_> = box B { x: box 0, y: box 1 };
    let _x = &mut a.x;
    //~^ NOTE borrow of `a.x` occurs here
    let _y = a.y; //~ ERROR cannot move
}

fn borrow_after_mut_borrow() {
    let mut a: Box<_> = box A { x: box 0, y: 1 };
    let _x = &mut a.x;
    //~^ NOTE previous borrow of `a` occurs here (through borrowing `a.x`);
    let _y = &a.y; //~ ERROR cannot borrow
}
//~^ NOTE previous borrow ends here

fn mut_borrow_after_borrow() {
    let mut a: Box<_> = box A { x: box 0, y: 1 };
    let _x = &a.x;
    //~^ NOTE previous borrow of `a` occurs here (through borrowing `a.x`)
    let _y = &mut a.y; //~ ERROR cannot borrow
}
//~^ NOTE previous borrow ends here

fn copy_after_move_nested() {
    let a: Box<_> = box C { x: box A { x: box 0, y: 1 }, y: 2 };
    let _x = a.x.x;
    //~^ NOTE `a.x.x` moved here because it has type `Box<isize>`, which is moved by default
    let _y = a.y; //~ ERROR use of collaterally moved
}

fn move_after_move_nested() {
    let a: Box<_> = box D { x: box A { x: box 0, y: 1 }, y: box 2 };
    let _x = a.x.x;
    //~^ NOTE `a.x.x` moved here because it has type `Box<isize>`, which is moved by default
    let _y = a.y; //~ ERROR use of collaterally moved
}

fn borrow_after_move_nested() {
    let a: Box<_> = box C { x: box A { x: box 0, y: 1 }, y: 2 };
    let _x = a.x.x;
    //~^ NOTE `a.x.x` moved here because it has type `Box<isize>`, which is moved by default
    let _y = &a.y; //~ ERROR use of collaterally moved
}

fn move_after_borrow_nested() {
    let a: Box<_> = box D { x: box A { x: box 0, y: 1 }, y: box 2 };
    let _x = &a.x.x;
    //~^ NOTE borrow of `a.x.x` occurs here
    let _y = a.y; //~ ERROR cannot move
}

fn copy_after_mut_borrow_nested() {
    let mut a: Box<_> = box C { x: box A { x: box 0, y: 1 }, y: 2 };
    let _x = &mut a.x.x;
    //~^ NOTE borrow of `a.x.x` occurs here
    let _y = a.y; //~ ERROR cannot use
}

fn move_after_mut_borrow_nested() {
    let mut a: Box<_> = box D { x: box A { x: box 0, y: 1 }, y: box 2 };
    let _x = &mut a.x.x;
    //~^ NOTE borrow of `a.x.x` occurs here
    let _y = a.y; //~ ERROR cannot move
}

fn borrow_after_mut_borrow_nested() {
    let mut a: Box<_> = box C { x: box A { x: box 0, y: 1 }, y: 2 };
    let _x = &mut a.x.x;
    //~^ NOTE previous borrow of `a.x.x` occurs here; the mutable borrow prevents
    let _y = &a.y; //~ ERROR cannot borrow
}
//~^ NOTE previous borrow ends here

fn mut_borrow_after_borrow_nested() {
    let mut a: Box<_> = box C { x: box A { x: box 0, y: 1 }, y: 2 };
    let _x = &a.x.x;
    //~^ NOTE previous borrow of `a.x.x` occurs here; the immutable borrow prevents
    let _y = &mut a.y; //~ ERROR cannot borrow
}
//~^ NOTE previous borrow ends here

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
