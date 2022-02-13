// check-pass

#![feature(ptr_metadata)]

fn a<T>() {
    b::<T>();
    b::<std::cell::Cell<T>>();
}

fn b<T: std::ptr::Pointee<Metadata = ()>>() {}

fn main() {}
