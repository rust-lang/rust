//@ check-pass
//@ edition:2021
//@ revisions: next old
//@[next] compile-flags: -Znext-solver=globally
//@[old] compile-flags: -Znext-solver=coherence

#![feature(unsize)]

use std::marker::{PhantomData, Unsize};

pub struct Ptr<T: ?Sized> {
    _phantom: PhantomData<T>,
}

impl<U: ?Sized> Ptr<U> {
    unsafe fn get(&self) -> *mut () {
        unimplemented!()
    }

    pub fn from_sized<T: Unsize<U>>(self, o: T) -> Self {
        unsafe {
            let data = self.get() as *mut T;
            std::ptr::write(data, o);
            todo!()
        }
    }
}

fn main() {}
