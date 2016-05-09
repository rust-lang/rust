#![feature(custom_attribute, box_syntax)]
#![allow(dead_code, unused_attributes)]

// error-pattern:can't handle destination layout StructWrappedNullablePointer

use std::cell::RefCell;
use std::rc::Rc;

struct Loop(Rc<RefCell<Option<Loop>>>);

#[miri_run]
fn rc_reference_cycle() -> Loop {
    let a = Rc::new(RefCell::new(None));
    let b = a.clone();
    *a.borrow_mut() = Some(Loop(b));
    Loop(a)
}

#[miri_run]
fn main() {
    let x = rc_reference_cycle().0;
    assert!(x.borrow().is_some());
}
