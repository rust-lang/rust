// run-pass
// compile-flags:-C debuginfo=2
// edition:2018

use core::marker::PhantomData;

pub struct Foo<T: ?Sized, A>(
    PhantomData<(A, T)>,
);

enum Never {}

impl<T: ?Sized> Foo<T, Never> {
    fn new_foo() -> Foo<T, Never> {
        Foo(PhantomData)
    }
}

fn main() {
    let _ = Foo::<[()], Never>::new_foo();
}
