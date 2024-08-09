//@ run-pass
#![allow(unused_variables)]
#![allow(non_upper_case_globals)]

// Test that destructor on a struct runs successfully after the struct
// is boxed and converted to an object.

use std::sync::atomic::{AtomicUsize, Ordering};

static value: AtomicUsize = AtomicUsize::new(0);

struct Cat {
    name : usize,
}

trait Dummy {
    fn get(&self) -> usize; //~ WARN method `get` is never used
}

impl Dummy for Cat {
    fn get(&self) -> usize { self.name }
}

impl Drop for Cat {
    fn drop(&mut self) {
        value.store(self.name, Ordering::Relaxed);
    }
}

pub fn main() {
    {
        let x = Box::new(Cat {name: 22});
        let nyan: Box<dyn Dummy> = x as Box<dyn Dummy>;
    }
    assert_eq!(value.load(Ordering::Relaxed), 22);
}
