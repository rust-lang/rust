//@ check-pass

use std::cell::RefCell;

pub fn f(v: Vec<RefCell<u8>>) {
    let _t = &mut *v[0].borrow_mut();
}

fn main() {}
