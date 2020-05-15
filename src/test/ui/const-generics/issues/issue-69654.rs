#![feature(const_generics)]
#![allow(incomplete_features)]

trait Bar<T> {}
impl<T> Bar<T> for [u8; T] {}
//~^ ERROR expected value, found type parameter `T`

struct Foo<const T: usize> {}
impl<const T: usize> Foo<T>
where
    [u8; T]: Bar<[(); T]>,
{
    fn foo() {}
}

fn main() {
    Foo::foo();
}
