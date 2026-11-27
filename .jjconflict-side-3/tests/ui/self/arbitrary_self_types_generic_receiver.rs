#![feature(arbitrary_self_types)]

struct PtrA<T>(T);

impl<T> core::ops::Receiver for PtrA<T> {
    type Target = T;
}

struct PtrB<T>(T);

trait SomePtr: core::ops::Receiver<Target=<Self as SomePtr>::SomeTarget> {
    type SomeTarget;
}

impl<T> SomePtr for PtrB<T> {
    type SomeTarget = T;
}

impl<T> core::ops::Receiver for PtrB<T> {
    type Target = T;
}

struct Content;

impl Content {
    fn a<R: core::ops::Receiver<Target=Self>>(self: &R) {}
    //~^ ERROR invalid generic
    fn b<R: core::ops::Receiver<Target=Self>>(self: &mut R) {}
    //~^ ERROR invalid generic
    fn c<R: core::ops::Receiver<Target=Self>>(self: R) {}
    //~^ ERROR invalid generic
    fn d<R: SomePtr<SomeTarget=Self>>(self: R) {}
    //~^ ERROR invalid generic
    fn e(self: impl SomePtr<SomeTarget=Self>) {}
    //~^ ERROR invalid generic
}

fn main() {
    PtrA(Content).a();
    PtrA(Content).b();
    PtrA(Content).c();
    std::rc::Rc::new(Content).a();
    std::rc::Rc::new(Content).b();
    std::rc::Rc::new(Content).c();
    PtrB(Content).a();
    PtrB(Content).b();
    PtrB(Content).c();
    PtrB(Content).d();
    PtrB(Content).e();
}
