#![feature(box_syntax)]

use std::thread;

fn borrow<T>(_: &T) { }

fn different_vars_after_borrows() {
    let x1: Box<_> = box 1;
    let p1 = &x1;
    let x2: Box<_> = box 2;
    let p2 = &x2;
    thread::spawn(move|| {
        drop(x1); //~ ERROR cannot move `x1` into closure because it is borrowed
        drop(x2); //~ ERROR cannot move `x2` into closure because it is borrowed
    });
    borrow(&*p1);
    borrow(&*p2);
}

fn different_vars_after_moves() {
    let x1: Box<_> = box 1;
    drop(x1);
    let x2: Box<_> = box 2;
    drop(x2);
    thread::spawn(move|| {
        drop(x1); //~ ERROR capture of moved value: `x1`
        drop(x2); //~ ERROR capture of moved value: `x2`
    });
}

fn same_var_after_borrow() {
    let x: Box<_> = box 1;
    let p = &x;
    thread::spawn(move|| {
        drop(x); //~ ERROR cannot move `x` into closure because it is borrowed
        drop(x); //~ ERROR use of moved value: `x`
    });
    borrow(&*p);
}

fn same_var_after_move() {
    let x: Box<_> = box 1;
    drop(x);
    thread::spawn(move|| {
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
