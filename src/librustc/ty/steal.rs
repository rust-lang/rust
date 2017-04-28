use std::cell::{Ref, RefCell};
use std::mem;

pub struct Steal<T> {
    value: RefCell<Option<T>>
}

impl<T> Steal<T> {
    pub fn new(value: T) -> Self {
        Steal {
            value: RefCell::new(Some(value))
        }
    }

    pub fn borrow(&self) -> Ref<T> {
        Ref::map(self.value.borrow(), |opt| match *opt {
            None => panic!("attempted to read from stolen value"),
            Some(ref v) => v
        })
    }

    pub fn steal(&self) -> T {
        let value_ref = &mut *self.value.borrow_mut();
        let value = mem::replace(value_ref, None);
        value.expect("attempt to read from stolen value")
    }
}
