//@ run-pass
#![feature(arbitrary_self_types)]
#![allow(unused_allocation)]

struct SmartPtr<T: ?Sized>(T);

impl<T: ?Sized> std::ops::Receiver for SmartPtr<T> {
    type Target = T;
}

trait Trait {
    fn trait_method<'a>(self: &'a Box<SmartPtr<Self>>) -> &'a [i32];
}

impl Trait for Vec<i32> {
    fn trait_method<'a>(self: &'a Box<SmartPtr<Self>>) -> &'a [i32] {
        &(**self).0
    }
}

fn main() {
    let v = vec![1, 2, 3];

    assert_eq!(&[1, 2, 3], Box::new(SmartPtr(v)).trait_method());
}
