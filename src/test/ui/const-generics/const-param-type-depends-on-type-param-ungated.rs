use std::marker::PhantomData;

struct B<T, const N: T>(PhantomData<[T; N]>); //~ ERROR const generics are unstable
//~^ ERROR the types of const generic parameters must derive `PartialEq` and `Eq`

fn main() {}
