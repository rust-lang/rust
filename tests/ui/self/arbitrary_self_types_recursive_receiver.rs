//@ run-pass
#![feature(arbitrary_self_types)]

struct MyNonNull<T>(*const T);

impl<T> std::ops::Receiver for MyNonNull<T> {
    type Target = T;
}

#[allow(dead_code)]
impl<T> MyNonNull<T> {
    fn foo<U>(&self) -> *const U {
        self.cast::<U>().bar()
    }
    fn cast<U>(&self) -> MyNonNull<U> {
        MyNonNull(self.0 as *const U)
    }
    fn bar(&self) -> *const T {
        self.0
    }
}

#[repr(transparent)]
struct Foo(usize);
#[repr(transparent)]
struct Bar(usize);

fn main() {
    let a = Foo(3);
    let ptr = MyNonNull(&a);
    let _bar_ptr: *const Bar = ptr.foo();
}
