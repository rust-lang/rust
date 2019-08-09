// build-pass (FIXME(62277): could be check-pass?)
use std::cell::RefCell;
use std::rc::Rc;

pub struct Callbacks {
    callbacks: Vec<Rc<RefCell<dyn FnMut(i32)>>>,
}

impl Callbacks {
    pub fn register<F: FnMut(i32)+'static>(&mut self, callback: F) {
        self.callbacks.push(Rc::new(RefCell::new(callback)));
    }
}

fn main() {}
