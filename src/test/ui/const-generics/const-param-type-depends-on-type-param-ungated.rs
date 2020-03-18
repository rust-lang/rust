#![feature(const_generics)]
//~^ WARNING the feature `const_generics` is incomplete and may cause the compiler to crash

use std::marker::PhantomData;

struct B<T, const N: T>(PhantomData<[T; N]>);
//~^ ERROR the types of const generic parameters must derive `PartialEq` and `Eq`

fn main() {}
