//@ run-pass
#![feature(derive_coerce_pointee, arbitrary_self_types)]

use std::marker::CoercePointee;

#[derive(CoercePointee)]
#[repr(transparent)]
struct MyPointer<'a, #[pointee] T: ?Sized> {
    ptr: &'a T,
}

impl<T: ?Sized> Copy for MyPointer<'_, T> {}
impl<T: ?Sized> Clone for MyPointer<'_, T> {
    fn clone(&self) -> Self {
        Self { ptr: self.ptr }
    }
}

impl<'a, T: ?Sized> core::ops::Deref for MyPointer<'a, T> {
    type Target = T;
    fn deref(&self) -> &'a T {
        self.ptr
    }
}

struct MyValue(u32);
impl MyValue {
    fn through_pointer(self: MyPointer<'_, Self>) -> u32 {
        self.ptr.0
    }
}

trait MyTrait {
    fn through_trait(&self) -> u32;
    fn through_trait_and_pointer(self: MyPointer<'_, Self>) -> u32;
}

impl MyTrait for MyValue {
    fn through_trait(&self) -> u32 {
        self.0
    }

    fn through_trait_and_pointer(self: MyPointer<'_, Self>) -> u32 {
        self.ptr.0
    }
}

pub fn main() {
    let v = MyValue(10);
    let ptr = MyPointer { ptr: &v };
    assert_eq!(v.0, ptr.through_pointer());
    assert_eq!(v.0, ptr.through_pointer());
    let dptr = ptr as MyPointer<dyn MyTrait>;
    assert_eq!(v.0, dptr.through_trait());
    assert_eq!(v.0, dptr.through_trait_and_pointer());
}
