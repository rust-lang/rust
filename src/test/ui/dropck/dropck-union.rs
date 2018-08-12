#![feature(untagged_unions)]

use std::cell::Cell;
use std::ops::Deref;
use std::mem::ManuallyDrop;

union Wrap<T> { x: ManuallyDrop<T> }

impl<T> Drop for Wrap<T>  {
    fn drop(&mut self) {
        unsafe { std::ptr::drop_in_place(&mut *self.x as *mut T); }
    }
}

impl<T> Wrap<T> {
    fn new(x: T) -> Self {
        Wrap { x: ManuallyDrop::new(x) }
    }
}

impl<T> Deref for Wrap<T> {
    type Target = T;
    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe {
            &self.x
        }
    }
}

struct C<'a>(Cell<Option<&'a C<'a>>>);

impl<'a> Drop for C<'a> {
    fn drop(&mut self) {}
}

fn main() {
    let v : Wrap<C> = Wrap::new(C(Cell::new(None)));
    v.0.set(Some(&v)); //~ ERROR: `v` does not live long enough
}
