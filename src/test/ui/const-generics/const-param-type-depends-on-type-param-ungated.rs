use std::marker::PhantomData;

struct B<T, const N: T>(PhantomData<[T; N]>); //~ ERROR const generics are unstable
//~^ ERROR `T` is not guaranteed to `#[derive(PartialEq, Eq)]`

fn main() {}
