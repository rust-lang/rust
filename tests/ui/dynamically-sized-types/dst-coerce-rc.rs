// run-pass
#![allow(unused_variables)]
#![allow(stable_features)]
// Test a very simple custom DST coercion.

#![feature(core, rc_weak)]

use std::cell::RefCell;
use std::rc::{Rc, Weak};

trait Baz {
    fn get(&self) -> i32;
}

impl Baz for i32 {
    fn get(&self) -> i32 {
        *self
    }
}

fn main() {
    let a: Rc<[i32; 3]> = Rc::new([1, 2, 3]);
    let b: Rc<[i32]> = a;
    assert_eq!(b[0], 1);
    assert_eq!(b[1], 2);
    assert_eq!(b[2], 3);

    let a: Rc<i32> = Rc::new(42);
    let b: Rc<dyn Baz> = a.clone();
    assert_eq!(b.get(), 42);

    let c: Weak<i32> = Rc::downgrade(&a);
    let d: Weak<dyn Baz> = c.clone();

    let _c = b.clone();

    let a: Rc<RefCell<i32>> = Rc::new(RefCell::new(42));
    let b: Rc<RefCell<dyn Baz>> = a.clone();
    assert_eq!(b.borrow().get(), 42);
    // FIXME
    let c: Weak<RefCell<dyn Baz>> = Rc::downgrade(&a) as Weak<_>;
}
