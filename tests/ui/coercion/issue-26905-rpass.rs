//@ run-pass
#![feature(unsize, coerce_unsized)]

// Verfies that PhantomData is ignored for DST coercions

use std::marker::{Unsize, PhantomData};
use std::ops::CoerceUnsized;

struct MyRc<T: ?Sized> {
    _ptr: *const T,
    _boo: PhantomData<T>,
}

impl<T: ?Sized + Unsize<U>, U: ?Sized> CoerceUnsized<MyRc<U>> for MyRc<T>{ }

fn main() {
    let data = [1, 2, 3];
    let iter = data.iter();
    let x = MyRc { _ptr: &iter, _boo: PhantomData };
    let _y: MyRc<dyn Iterator<Item=&u32>> = x;
}
