//@ run-pass
use std::ops::{Deref, DerefMut};

pub struct MyRef(u32);

impl Deref for MyRef {
    type Target = u32;
    fn deref(&self) -> &u32 { &self.0 }
}

impl DerefMut for MyRef {
    fn deref_mut(&mut self) -> &mut u32 { &mut self.0 }
}


fn deref(x: &MyRef) -> &u32 {
    x
}

fn deref_mut(x: &mut MyRef) -> &mut u32 {
    x
}

fn main() {
    let mut r = MyRef(2);
    assert_eq!(deref(&r) as *const _, &r.0 as *const _);
    assert_eq!(deref_mut(&mut r) as *mut _, &mut r.0 as *mut _);
}
