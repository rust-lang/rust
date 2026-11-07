//! Regression test for issue #20126: Copy and Drop traits are mutually exclusive

#[derive(Copy, Clone)]
struct Foo; //~ ERROR the trait `Copy` cannot be implemented

impl Drop for Foo {
    fn drop(&mut self) {}
}

#[derive(Copy, Clone)]
struct Bar<T>(::std::marker::PhantomData<T>); //~ ERROR the trait `Copy` cannot be implemented

impl<T> Drop for Bar<T> {
    fn drop(&mut self) {}
}

fn main() {}
