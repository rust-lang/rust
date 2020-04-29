#![feature(const_generics)]
#![allow(incomplete_features)]

trait Bar<O> {}
impl<O> Bar<O> for [u8; O] {}
//~^ ERROR expected value, found type parameter `O`

struct Foo<const O: usize> {}
impl<const O: usize> Foo<O>
where
    [u8; O]: Bar<[(); O]>,
{
    fn foo() {}
}

fn main() {
    Foo::foo();
}
