// run-pass

#![feature(ptr_internals)]

use std::ptr::Unique;

const PTR: *mut u32 = Unique::empty().as_ptr();

fn ident<T>(ident: T) -> T {
    ident
}

pub fn main() {
    assert_eq!(PTR, ident(Unique::<u32>::empty().as_ptr()));
}
