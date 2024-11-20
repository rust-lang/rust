struct SmartPtr<'a, T: ?Sized>(&'a T);

impl<T> std::ops::Receiver for SmartPtr<'_, T> {
    type Target = T;
}

struct A;

trait B {
    fn m(self: SmartPtr<Self>) {}
    //~^ ERROR: invalid `self` parameter type
}

impl B for A {
    fn m(self: SmartPtr<Self>) {}
}

fn main() {
    let a = A;
    let a = SmartPtr(&A);
    a.m();
}
