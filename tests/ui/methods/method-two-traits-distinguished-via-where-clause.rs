//@ run-pass
// Test that we select between traits A and B. To do that, we must
// consider the `Sized` bound.


trait A { //~ WARN trait `A` is never used
    fn foo(self);
}

trait B {
    fn foo(self);
}

impl<T: Sized> A for *const T {
    fn foo(self) {}
}

impl<T> B for *const [T] {
    fn foo(self) {}
}

fn main() {
    let x: [isize; 4] = [1,2,3,4];
    let xptr = &x[..] as *const [isize];
    xptr.foo();
}
