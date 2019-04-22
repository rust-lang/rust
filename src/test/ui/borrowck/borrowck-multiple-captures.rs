#![feature(box_syntax)]

use std::thread;

fn borrow<T>(_: &T) { }

fn different_vars_after_borrows() {
    let x1: Box<_> = box 1;
    let p1 = &x1;
    let x2: Box<_> = box 2;
    let p2 = &x2;
    thread::spawn(move|| {
        //~^ ERROR cannot move out of `x1` because it is borrowed
        //~| ERROR cannot move out of `x2` because it is borrowed
        drop(x1);
        drop(x2);
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
        //~^ ERROR use of moved value: `x1`
        //~| ERROR use of moved value: `x2`
        drop(x1);
        drop(x2);
    });
}

fn same_var_after_borrow() {
    let x: Box<_> = box 1;
    let p = &x;
    thread::spawn(move|| {
        //~^ ERROR cannot move out of `x` because it is borrowed
        drop(x);
        drop(x); //~ ERROR use of moved value: `x`
    });
    borrow(&*p);
}

fn same_var_after_move() {
    let x: Box<_> = box 1;
    drop(x);
    thread::spawn(move|| {
        //~^ ERROR use of moved value: `x`
        drop(x);
        drop(x); //~ ERROR use of moved value: `x`
    });
}

fn main() {
    different_vars_after_borrows();
    different_vars_after_moves();
    same_var_after_borrow();
    same_var_after_move();
}
