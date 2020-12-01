#![feature(const_generics)]
#![allow(incomplete_features)]

trait Bar<T> {}
impl<T> Bar<T> for [u8; T] {}
//~^ ERROR expected value, found type parameter `T`

struct Foo<const N: usize> {}
impl<const N: usize> Foo<N>
where
    [u8; N]: Bar<[(); N]>,
{
    fn foo() {}
}

fn main() {
    Foo::foo();
}
