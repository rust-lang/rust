#![feature(unsize, coerce_unsized)]
#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]

// Verfies that non-PhantomData ZSTs still cause coercions to fail.
// They might have additional semantics that we don't want to bulldoze.

use std::marker::{Unsize, PhantomData};
use std::ops::CoerceUnsized;

struct NotPhantomData<T>(PhantomData<T>);

struct MyRc<T> {
    _ptr: *const T,
    _boo: NotPhantomData<T>,
}

impl<T: Unsize<U>, U> CoerceUnsized<MyRc<U>> for MyRc<T>{ } //~ERROR

fn main() {
    let data = [1, 2, 3];
    let iter = data.iter();
    let x = MyRc { _ptr: &iter, _boo: NotPhantomData(PhantomData) };
    let _y: MyRc<dyn Iterator<Item=&u32>> = x;
}
