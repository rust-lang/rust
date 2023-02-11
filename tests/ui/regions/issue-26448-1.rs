// run-pass

pub trait Foo<T> {
    fn foo(self) -> T;
}

impl<'a, T> Foo<T> for &'a str where &'a str: Into<T> {
    fn foo(self) -> T {
        panic!();
    }
}

fn main() {}
