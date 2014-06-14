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

fn field_copy_after_field_borrow() {
    let mut x = A { a: 1, b: box 2 };
    let p = &mut x.b;
    drop(x.a);
    **p = 3;
}

fn fu_field_copy_after_field_borrow() {
    let mut x = A { a: 1, b: box 2 };
    let p = &mut x.b;
    let y = A { b: box 3, .. x };
    drop(y);
    **p = 4;
}

fn field_deref_after_field_borrow() {
    let mut x = A { a: 1, b: box 2 };
    let p = &mut x.a;
    drop(*x.b);
    *p = 3;
}

fn field_move_after_field_borrow() {
    let mut x = A { a: 1, b: box 2 };
    let p = &mut x.a;
    drop(x.b);
    *p = 3;
}

fn fu_field_move_after_field_borrow() {
    let mut x = A { a: 1, b: box 2 };
    let p = &mut x.a;
    let y = A { a: 3, .. x };
    drop(y);
    *p = 4;
}

fn main() {
    field_copy_after_field_borrow();
    fu_field_copy_after_field_borrow();
    field_deref_after_field_borrow();
    field_move_after_field_borrow();
    fu_field_move_after_field_borrow();
}

