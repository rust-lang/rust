#![feature(const_generics)]
#![allow(incomplete_features, unused_braces)]

trait Bar<T> {}
impl<T> Bar<T> for [u8; {7}] {}

struct Foo<const N: usize> {}
impl<const N: usize> Foo<N>
where
    [u8; N]: Bar<[(); N]>,
{
    fn foo() {}
}

fn main() {
    Foo::foo();
    //~^ ERROR the function or associated item
}
