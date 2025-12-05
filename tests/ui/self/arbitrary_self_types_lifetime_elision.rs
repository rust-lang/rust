//@ run-pass

#![feature(arbitrary_self_types)]

#[derive(Clone)]
struct SmartPtr<'a, T: ?Sized>(&'a T);

impl<'a, T: ?Sized> std::ops::Receiver for SmartPtr<'a, T> {
    type Target = T;
}

#[derive(Clone)]
struct MyType;

impl MyType {
    fn m(self: SmartPtr<Self>) {}
    fn n(self: SmartPtr<'_, Self>) {}
    fn o<'a>(self: SmartPtr<'a, Self>) {}
}

fn main() {
    let a = MyType;
    let ptr = SmartPtr(&a);
    ptr.clone().m();
    ptr.clone().n();
    ptr.o();
}
