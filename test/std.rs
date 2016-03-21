#![feature(custom_attribute, box_syntax)]
#![allow(dead_code, unused_attributes)]

#[miri_run]
fn rc_cell() -> i32 {
    use std::rc::Rc;
    use std::cell::Cell;
    let r = Rc::new(Cell::new(42));
    let x = r.get();
    r.set(x + x);
    r.get()
}

#[miri_run]
fn arc() -> i32 {
    use std::sync::Arc;
    let a = Arc::new(42);
    *a
}

#[miri_run]
fn true_assert() {
    assert_eq!(1, 1);
}
