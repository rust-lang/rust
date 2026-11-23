//@ known-bug: #145383
//@ edition: 2024

#![expect(incomplete_features)]
#![feature(field_projections)]

use std::ops::place::raw_handle;

struct Pair {
    a: String,
    b: String,
}

fn disjoint_mut_borrow() {
    let mut s = Pair { a: String::new(), b: String::new() };
    let r = &mut s.a;
    let mut c = || unsafe { raw_handle!(s.b) };
    let _ = c();
    r.push('x');
}

fn disjoint_move_while_borrowed() {
    let s = Pair { a: String::new(), b: String::new() };
    let c = || unsafe { raw_handle!(s.b) };
    drop(s.a);
    drop(c);
}

fn borrow_after_partial_move() {
    let mut s = Pair { a: String::new(), b: String::new() };
    drop(s.a);
    let mut c = || unsafe { raw_handle!(s.b) };
    let _ = c();
}

fn main() {
    disjoint_mut_borrow();
    disjoint_move_while_borrowed();
    borrow_after_partial_move();
}
