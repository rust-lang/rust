// compile-flags: -Zsave-analysis
// Regression test for #69414 ^

use std::marker::PhantomData;

struct B<T, const N: T>(PhantomData<[T; N]>);
//~^ ERROR the type of const parameters must not depend on other generic parameters

fn main() {}
