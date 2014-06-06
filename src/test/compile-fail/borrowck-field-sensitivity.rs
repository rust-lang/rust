// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct A { a: int, b: Box<int> }

fn borrow<T>(_: &T) { }

fn use_after_move() {
    let x = A { a: 1, b: box 2 };
    drop(x.b);
    drop(*x.b); //~ ERROR use of partially moved value: `*x.b`
}

fn use_after_fu_move() {
    let x = A { a: 1, b: box 2 };
    let y = A { a: 3, .. x };
    drop(*x.b); //~ ERROR use of partially moved value: `*x.b`
}

fn borrow_after_move() {
    let x = A { a: 1, b: box 2 };
    drop(x.b);
    borrow(&x.b); //~ ERROR use of moved value: `x.b`
}

fn borrow_after_fu_move() {
    let x = A { a: 1, b: box 2 };
    let _y = A { a: 3, .. x };
    borrow(&x.b); //~ ERROR use of moved value: `x.b`
}

fn move_after_borrow() {
    let x = A { a: 1, b: box 2 };
    let y = &x.b;
    drop(x.b); //~ ERROR cannot move out of `x.b` because it is borrowed
    borrow(&*y);
}

fn fu_move_after_borrow() {
    let x = A { a: 1, b: box 2 };
    let y = &x.b;
    let _z = A { a: 3, .. x }; //~ ERROR cannot move out of `x.b` because it is borrowed
    borrow(&*y);
}

fn mut_borrow_after_mut_borrow() {
    let mut x = A { a: 1, b: box 2 };
    let y = &mut x.a;
    let z = &mut x.a; //~ ERROR cannot borrow `x.a` as mutable more than once at a time
    drop(*y);
    drop(*z);
}

fn move_after_move() {
    let x = A { a: 1, b: box 2 };
    drop(x.b);
    drop(x.b);  //~ ERROR use of moved value: `x.b`
}

fn move_after_fu_move() {
    let x = A { a: 1, b: box 2 };
    let _y = A { a: 3, .. x };
    drop(x.b);  //~ ERROR use of moved value: `x.b`
}

fn fu_move_after_move() {
    let x = A { a: 1, b: box 2 };
    drop(x.b);
    let _z = A { a: 3, .. x };  //~ ERROR use of moved value: `x.b`
}

fn fu_move_after_fu_move() {
    let x = A { a: 1, b: box 2 };
    let _y = A { a: 3, .. x };
    let _z = A { a: 4, .. x };  //~ ERROR use of moved value: `x.b`
}

// The following functions aren't yet accepted, but they should be.

fn use_after_field_assign_after_uninit() {
    let mut x: A;
    x.a = 1;
    drop(x.a); //~ ERROR use of possibly uninitialized variable: `x.a`
}

fn borrow_after_field_assign_after_uninit() {
    let mut x: A;
    x.a = 1;
    borrow(&x.a); //~ ERROR use of possibly uninitialized variable: `x.a`
}

fn move_after_field_assign_after_uninit() {
    let mut x: A;
    x.b = box 1;
    drop(x.b); //~ ERROR use of possibly uninitialized variable: `x.b`
}

fn main() {
    use_after_move();
    use_after_fu_move();

    borrow_after_move();
    borrow_after_fu_move();
    move_after_borrow();
    fu_move_after_borrow();
    mut_borrow_after_mut_borrow();

    move_after_move();
    move_after_fu_move();
    fu_move_after_move();
    fu_move_after_fu_move();

    use_after_field_assign_after_uninit();
    borrow_after_field_assign_after_uninit();
    move_after_field_assign_after_uninit();
}

