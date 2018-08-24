// issue #20126

#[derive(Copy, Clone)] //~ ERROR the trait `Copy` may not be implemented
struct Foo;

impl Drop for Foo {
    fn drop(&mut self) {}
}

#[derive(Copy, Clone)] //~ ERROR the trait `Copy` may not be implemented
struct Bar<T>(::std::marker::PhantomData<T>);

impl<T> Drop for Bar<T> {
    fn drop(&mut self) {}
}

fn main() {}
