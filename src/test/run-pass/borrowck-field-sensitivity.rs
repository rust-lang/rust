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
struct B { a: Box<int>, b: Box<int> }

fn borrow<T>(_: &T) { }

fn move_after_use() {
    let x = A { a: 1, b: box 2 };
    drop(x.a);
    drop(x.b);
}

fn move_after_fu_use() {
    let x = A { a: 1, b: box 2 };
    let _y = A { b: box 3, .. x };
    drop(x.b);
}

fn fu_move_after_use() {
    let x = A { a: 1, b: box 2 };
    drop(x.a);
    let y = A { a: 3, .. x };
    drop(y.b);
}

fn fu_move_after_fu_use() {
    let x = A { a: 1, b: box 2 };
    let _y = A { b: box 3, .. x };
    let z = A { a: 4, .. x };
    drop(z.b);
}

fn use_after_move() {
    let x = A { a: 1, b: box 2 };
    drop(x.b);
    drop(x.a);
}

fn use_after_fu_move() {
    let x = A { a: 1, b: box 2 };
    let y = A { a: 3, .. x };
    drop(x.a);
    drop(y.b);
}

fn fu_use_after_move() {
    let x = A { a: 1, b: box 2 };
    drop(x.b);
    let _y = A { b: box 3, .. x };
}

fn fu_use_after_fu_move() {
    let x = A { a: 1, b: box 2 };
    let y = A { a: 3, .. x };
    let _z = A { b: box 3, .. x };
    drop(y.b);
}

fn borrow_after_move() {
    let x = A { a: 1, b: box 2 };
    drop(x.b);
    borrow(&x.a);
}

fn borrow_after_fu_move() {
    let x = A { a: 1, b: box 2 };
    let y = A { a: 3, .. x };
    borrow(&x.a);
    drop(y.b);
}

fn move_after_borrow() {
    let x = A { a: 1, b: box 2 };
    borrow(&x.a);
    drop(x.b);
}

fn fu_move_after_borrow() {
    let x = A { a: 1, b: box 2 };
    borrow(&x.a);
    let y = A { a: 3, .. x };
    drop(y.b);
}

fn mut_borrow_after_mut_borrow() {
    let mut x = A { a: 1, b: box 2 };
    let y = &mut x.a;
    let z = &mut x.b;
    drop(*y);
    drop(**z);
}

fn move_after_move() {
    let x = B { a: box 1, b: box 2 };
    drop(x.a);
    drop(x.b);
}

fn move_after_fu_move() {
    let x = B { a: box 1, b: box 2 };
    let y = B { a: box 3, .. x };
    drop(x.a);
    drop(y.b);
}

fn fu_move_after_move() {
    let x = B { a: box 1, b: box 2 };
    drop(x.a);
    let z = B { a: box 3, .. x };
    drop(z.b);
}

fn fu_move_after_fu_move() {
    let x = B { a: box 1, b: box 2 };
    let y = B { b: box 3, .. x };
    let z = B { a: box 4, .. x };
    drop(y.a);
    drop(z.b);
}

fn use_after_assign_after_move() {
    let mut x = A { a: 1, b: box 2 };
    drop(x.b);
    x = A { a: 3, b: box 4 };
    drop(*x.b);
}

fn use_after_field_assign_after_move() {
    let mut x = A { a: 1, b: box 2 };
    drop(x.b);
    x.b = box 3;
    drop(*x.b);
}

fn borrow_after_assign_after_move() {
    let mut x = A { a: 1, b: box 2 };
    drop(x.b);
    x = A { a: 3, b: box 4 };
    borrow(&x.b);
}

fn borrow_after_field_assign_after_move() {
    let mut x = A { a: 1, b: box 2 };
    drop(x.b);
    x.b = box 3;
    borrow(&x.b);
}

fn move_after_assign_after_move() {
    let mut x = A { a: 1, b: box 2 };
    let y = x.b;
    x = A { a: 3, b: box 4 };
    drop(x.b);
    drop(y);
}

fn move_after_field_assign_after_move() {
    let mut x = A { a: 1, b: box 2 };
    drop(x.b);
    x.b = box 3;
    drop(x.b);
}

fn use_after_assign_after_uninit() {
    let mut x: A;
    x = A { a: 1, b: box 2 };
    drop(x.a);
}

fn borrow_after_assign_after_uninit() {
    let mut x: A;
    x = A { a: 1, b: box 2 };
    borrow(&x.a);
}

fn move_after_assign_after_uninit() {
    let mut x: A;
    x = A { a: 1, b: box 2 };
    drop(x.b);
}

fn main() {
    move_after_use();
    move_after_fu_use();
    fu_move_after_use();
    fu_move_after_fu_use();
    use_after_move();
    use_after_fu_move();
    fu_use_after_move();
    fu_use_after_fu_move();

    borrow_after_move();
    borrow_after_fu_move();
    move_after_borrow();
    fu_move_after_borrow();
    mut_borrow_after_mut_borrow();

    move_after_move();
    move_after_fu_move();
    fu_move_after_move();
    fu_move_after_fu_move();

    use_after_assign_after_move();
    use_after_field_assign_after_move();
    borrow_after_assign_after_move();
    borrow_after_field_assign_after_move();
    move_after_assign_after_move();
    move_after_field_assign_after_move();

    use_after_assign_after_uninit();
    borrow_after_assign_after_uninit();
    move_after_assign_after_uninit();
}

