// #99269

//@ check-pass

#![warn(unused_must_use)]

unsafe fn free<T>(ptr: *mut T) {
    Box::from_raw(ptr); //~ WARNING unused return value
}

fn main() {}
