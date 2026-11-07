//@ run-pass

#![feature(share_trait)]

use std::cell::Cell;
use std::clone::Share;
use std::rc::Rc;

trait Value {
    fn get(&self) -> i32;
}

impl Value for Cell<i32> {
    fn get(&self) -> i32 {
        Cell::get(self)
    }
}

fn main() {
    let value = Rc::new(Cell::new(1));
    let shared = value.share();

    assert!(Rc::ptr_eq(&value, &shared));
    shared.set(2);
    assert_eq!(value.get(), 2);

    let dyn_value: Rc<dyn Value> = Rc::new(Cell::new(3));
    let shared_dyn_value = dyn_value.share();

    assert!(Rc::ptr_eq(&dyn_value, &shared_dyn_value));
    assert_eq!(shared_dyn_value.get(), 3);
}
