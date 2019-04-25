use std::marker::PhantomData;

struct B<T = N, const N: T>(PhantomData<[T; N]>);
//~^ ERROR expected type, found const parameter `N`
//~| ERROR const generics are unstable
//~| ERROR mismatched types

fn main() {}
