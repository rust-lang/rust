#![feature(custom_attribute, box_syntax)]
#![allow(dead_code, unused_attributes)]

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

#[miri_run]
fn rc_cell() -> Rc<Cell<i32>> {
    let r = Rc::new(Cell::new(42));
    let x = r.get();
    r.set(x + x);
    r
}

// TODO(tsion): borrow code needs to evaluate string statics via Lvalue::Static
// TODO(tsion): also requires destructors to run for the second borrow to work
// #[miri_run]
// fn rc_refcell() -> i32 {
//     let r = Rc::new(RefCell::new(42));
//     *r.borrow_mut() += 10;
//     let x = *r.borrow();
//     x
// }

#[miri_run]
fn arc() -> Arc<i32> {
    let a = Arc::new(42);
    a
}

#[miri_run]
fn true_assert() {
    assert_eq!(1, 1);
}

#[miri_run]
fn main() {
    assert_eq!(*arc(), 42);
    assert_eq!(rc_cell().get(), 84);
}
