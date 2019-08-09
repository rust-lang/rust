#![crate_type="lib"]

use std::cell::RefCell;

pub struct Window<Data>{
    pub data: RefCell<Data>
}

impl<Data:  Update> Window<Data> {
    pub fn update(&self, e: i32) {
        match e {
            1 => self.data.borrow_mut().update(),
            _ => {}
        }
    }
}

pub trait Update {
    fn update(&mut self);
}
