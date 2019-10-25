use std::marker::PhantomData;

struct B<T, const N: T>(PhantomData<[T; N]>); //~ ERROR const generics are unstable
//~^ ERROR const parameters cannot depend on type parameters

fn main() {}
